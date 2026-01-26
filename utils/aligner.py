import torch
from tqdm import tqdm

class StaticTokenAligner:
    """
    Optimized Aligner: Pre-computes the mapping table on CPU once, 
    then performs O(1) lookups on GPU during training.
    """
    def __init__(self, src_tokenizer, tgt_tokenizer, device="cuda"):
        self.device = device
        print("Pre-computing Token Alignment Tables (This takes ~30s)...")
        # Src -> Tgt (For Logits: Guidance -> Backbone)
        self.logit_map = self._build_mapping(src_tokenizer, tgt_tokenizer, "Aligning Logits")
        # Tgt -> Src (For Inputs: Backbone -> Guidance)
        self.input_map = self._build_mapping(tgt_tokenizer, src_tokenizer, "Aligning Inputs")
        
    def _build_mapping(self, src_tok, tgt_tok, desc):
        # 1. Create a tensor filled with -1 (meaning "no mapping")
        vocab_size = src_tok.vocab_size
        mapping = torch.full((vocab_size,), -1, dtype=torch.long)
        
        # 2. Iterate over the entire vocabulary once
        src_vocab = src_tok.get_vocab()
        
        # We perform string processing on CPU here (Offline phase)
        for token, src_id in tqdm(src_vocab.items(), desc=desc):
            if src_id >= vocab_size: continue
            
            # Handle special tokens carefully
            if token in src_tok.all_special_tokens:
                continue 
                
            # Decode -> Encode
            # Note: handle the "Ġ" prefix for BPE tokenizers by converting to string properly
            text = src_tok.convert_tokens_to_string([token])
            
            # Try to find exact match in target
            tgt_ids = tgt_tok.encode(text, add_special_tokens=False)
            
            # Only accept 1-to-1 mappings for simplicity and speed
            if len(tgt_ids) == 1:
                mapping[src_id] = tgt_ids[0]
                
        # Move the table to GPU memory
        return mapping.to(self.device)

    def translate_input(self, tgt_ids: torch.Tensor, return_attention_mask: bool = False):
        """
        Maps Backbone IDs (Tgt) -> Guidance IDs (Src) for input.
        """
        max_id = self.input_map.size(0)
        valid_mask = (tgt_ids < max_id)
        safe_ids = torch.where(valid_mask, tgt_ids, torch.tensor(0, device=self.device))
        
        mapped_ids = self.input_map[safe_ids]
        
        # If no mapping (-1) or invalid input, use 0 (UNK/PAD assumption)
        out_ids = torch.where((mapped_ids != -1) & valid_mask, mapped_ids, torch.tensor(0, device=self.device))
        if return_attention_mask:
            attn = torch.ones_like(out_ids, dtype=torch.long, device=self.device)
            return out_ids, attn
        return out_ids

    def align(self, src_logits: torch.Tensor, tgt_vocab_size: int, topk: int = 50) -> torch.Tensor:
        """
        Pure GPU operation.
        src_logits: [batch, src_vocab]
        """
        # 1. Get Top-K from Source (GPU)
        probs = torch.softmax(src_logits, dim=-1)
        topk_probs, topk_src_ids = probs.topk(topk, dim=-1)
        
        # [FIX] Guard against src_ids out of bounds for the mapping table
        max_src_id = self.logit_map.size(0)
        valid_src_mask = (topk_src_ids < max_src_id)
        
        # Use a safe index for lookup where invalid (index 0), we will filter later using valid_src_mask
        safe_src_ids = torch.where(valid_src_mask, topk_src_ids, torch.tensor(0, device=self.device))
        
        # 2. Fast Lookup via Tensor Indexing (GPU)
        topk_tgt_ids = self.logit_map[safe_src_ids]
        
        # 3. Create Target Logits (Sparse)
        # Initialize with -inf
        batch_size = src_logits.size(0)
        tgt_logits = torch.full((batch_size, tgt_vocab_size), float('-inf'), device=self.device, dtype=src_logits.dtype)
        
        # 4. Scatter values into target logits
        # We need to filter out:
        # - -1 (no valid mapping)
        # - IDs >= tgt_vocab_size (out of bounds for target)
        # - Invalid source IDs (out of bounds for mapping table)
        valid_mask = (topk_tgt_ids != -1) & (topk_tgt_ids < tgt_vocab_size) & valid_src_mask
        
        # Since we usually run with batch_size=1 during rollout, optimized path:
        if batch_size == 1:
            flat_indices = topk_tgt_ids[valid_mask]
            flat_values = topk_probs[valid_mask]
            
            # Log space for logits
            log_vals = torch.log(flat_values + 1e-10)
            
            # Scatter max to handle collisions
            if flat_indices.numel() > 0:
                tgt_logits[0].scatter_reduce_(0, flat_indices, log_vals, reduce='max', include_self=False)
        else:
            # General batch implementation
            for b in range(batch_size):
                mask_b = valid_mask[b]
                if mask_b.any():
                    idx = topk_tgt_ids[b][mask_b]
                    val = torch.log(topk_probs[b][mask_b] + 1e-10)
                    tgt_logits[b].scatter_reduce_(0, idx, val, reduce='max', include_self=False)

        return tgt_logits

class RobustTokenAligner:
    def __init__(self, src_tokenizer, tgt_tokenizer, device="cuda"):
        """
        src_tokenizer: Guidance Model Tokenizer
        tgt_tokenizer: Backbone Model Tokenizer
        """
        self.device = device
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        
        print("Building Token Alignment Table (full target tokenization)...")
        # Store full target tokenization per source id
        self.logit_map = self._build_full_mapping()

    def _build_full_mapping(self):
        # Map: Source_ID -> List[Target_ID]
        vocab_size = self.src_tokenizer.vocab_size
        mapping = [None] * vocab_size
        
        src_vocab = self.src_tokenizer.get_vocab()
        
        for token, src_id in tqdm(src_vocab.items(), desc="Aligning Logits"):
            if src_id >= vocab_size: continue
            
            # Decode the source token to text
            text = self.src_tokenizer.convert_tokens_to_string([token])
            
            # Encode with target tokenizer
            tgt_ids = self.tgt_tokenizer.encode(text, add_special_tokens=False)
            
            if len(tgt_ids) > 0:
                mapping[src_id] = tgt_ids

        return mapping

    def translate_input(self, tgt_ids: torch.Tensor, return_attention_mask: bool = False):
        """
        Robust Input Translation (Backbone IDs -> Guidance IDs).
        Uses Decode -> Encode to handle token splitting correctly.
        """
        # 1. Decode Backbone IDs to Text
        # We process the batch on CPU because Tokenizers aren't GPU compatible usually
        tgt_ids_list = tgt_ids.tolist()
        
        # If batch size is 1 (common in rollout), this is fast
        text_batch = self.tgt_tokenizer.batch_decode(tgt_ids_list, skip_special_tokens=True)
        
        # 2. Re-encode using Guidance Tokenizer
        encodings = self.src_tokenizer(text_batch, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = encodings.input_ids.to(self.device)
        attention_mask = encodings.attention_mask.to(self.device)
        return (input_ids, attention_mask) if return_attention_mask else input_ids

    def align_logits(self, src_logits: torch.Tensor, tgt_vocab_size: int, topk: int = 50) -> torch.Tensor:
        """
        Projects Guidance Logits -> Backbone Vocabulary.
        """
        # 1. Top-K on Guidance (Source)
        probs = torch.softmax(src_logits, dim=-1)
        topk_probs, topk_src_ids = probs.topk(topk, dim=-1)
        
        # 2. Scatter into Target Logits using full target tokenization
        batch_size = src_logits.size(0)
        tgt_logits = torch.full((batch_size, tgt_vocab_size), float('-inf'), device=self.device, dtype=src_logits.dtype)

        for b in range(batch_size):
            for k in range(topk_src_ids.size(1)):
                src_id = int(topk_src_ids[b, k].item())
                if src_id < 0 or src_id >= len(self.logit_map):
                    continue
                tgt_ids = self.logit_map[src_id]
                if not tgt_ids:
                    continue
                log_val = torch.log(topk_probs[b, k] + 1e-10)
                # Scatter to all target tokens produced by this source token
                for tgt_id in tgt_ids:
                    if 0 <= tgt_id < tgt_vocab_size:
                        tgt_logits[b].scatter_reduce_(0, torch.tensor([tgt_id], device=self.device), log_val, reduce='max', include_self=False)

        return tgt_logits

    def align(self, src_logits: torch.Tensor, tgt_vocab_size: int, topk: int = 50) -> torch.Tensor:
        return self.align_logits(src_logits, tgt_vocab_size, topk=topk)

class CachedTokenAligner:
    """
    Zero-Latency Aligner.
    Pre-computes ALL token translations into a GPU Lookup Table.
    Eliminates CPU String operations during training.
    """
    def __init__(self, src_tokenizer, tgt_tokenizer, device="cuda", max_fragments=4, max_tgt_tokens=4):
        self.device = device
        self.src_tokenizer = src_tokenizer # Guidance
        self.tgt_tokenizer = tgt_tokenizer # Backbone
        self.max_fragments = max_fragments # Max tokens a single backbone token splits into
        self.max_tgt_tokens = max_tgt_tokens # Max target tokens per guidance token
        
        print("Pre-computing Full Vocabulary Translation (GPU Cached)...")
        
        # 1. Build Input Map: Backbone ID -> [Guidance ID 1, Guidance ID 2, ...]
        # This replaces the slow "translate_input"
        self.input_map, self.input_lens = self._build_input_cache()
        
        # 2. Build Logit Map: Guidance ID -> Backbone IDs (Full Tokenization)
        self.logit_map_ids, self.logit_map_lens = self._build_logit_cache()
        
    def _build_input_cache(self):
        """
        For every token in Backbone Vocab, decode it and re-encode with Guidance.
        Store as (Vocab_Size, Max_Fragments) tensor on GPU.
        """
        vocab_size = self.tgt_tokenizer.vocab_size
        
        # Storage: Fill with Pad Token or 0
        cache = torch.full((vocab_size, self.max_fragments), 0, dtype=torch.long)
        lens = torch.zeros((vocab_size,), dtype=torch.long)
        
        tgt_vocab = self.tgt_tokenizer.get_vocab()
        
        for token, tgt_id in tqdm(tgt_vocab.items(), desc="Caching Input Map"):
            if tgt_id >= vocab_size: continue
            
            # 1. Decode Backbone Token -> String
            text = self.tgt_tokenizer.convert_tokens_to_string([token])
            
            # 2. Encode String -> Guidance Tokens
            # add_special_tokens=False is crucial here
            src_ids = self.src_tokenizer.encode(text, add_special_tokens=False)
            
            # 3. Store in Tensor
            length = min(len(src_ids), self.max_fragments)
            if length > 0:
                cache[tgt_id, :length] = torch.tensor(src_ids[:length])
                lens[tgt_id] = length
                
        return cache.to(self.device), lens.to(self.device)

    def _build_logit_cache(self):
        """
        Guidance -> Backbone (Full Target Tokenization)
        """
        vocab_size = self.src_tokenizer.vocab_size
        ids = torch.full((vocab_size, self.max_tgt_tokens), 0, dtype=torch.long)
        lens = torch.zeros((vocab_size,), dtype=torch.long)
        src_vocab = self.src_tokenizer.get_vocab()
        
        for token, src_id in tqdm(src_vocab.items(), desc="Caching Logit Map"):
            if src_id >= vocab_size: continue
            text = self.src_tokenizer.convert_tokens_to_string([token])
            tgt_ids = self.tgt_tokenizer.encode(text, add_special_tokens=False)
            if len(tgt_ids) > 0:
                length = min(len(tgt_ids), self.max_tgt_tokens)
                ids[src_id, :length] = torch.tensor(tgt_ids[:length], dtype=torch.long)
                lens[src_id] = length
                
        return ids.to(self.device), lens.to(self.device)

    def translate_input(self, tgt_ids: torch.Tensor, return_attention_mask: bool = False):
        """
        PURE GPU Operation.
        Maps Backbone Context -> Guidance Context.
        """
        # tgt_ids shape: [Batch, Seq_Len]

        # 1. Look up fragments safely: [Batch, Seq_Len, Max_Fragments]
        max_id = self.input_map.size(0)
        valid_mask_ids = (tgt_ids >= 0) & (tgt_ids < max_id)
        safe_ids = torch.where(valid_mask_ids, tgt_ids, torch.tensor(0, device=self.device))
        fragments = self.input_map[safe_ids]
        lengths = self.input_lens[safe_ids]
        lengths = torch.where(valid_mask_ids, lengths, torch.zeros_like(lengths))
        
        # 2. Flatten and filter padding
        # This effectively "stitches" the new sequence together
        # We use a mask to keep only valid fragments
        B, L, K = fragments.shape
        
        # Create a boolean mask of valid tokens: [B, L, K]
        mask = (torch.arange(K, device=self.device).view(1, 1, K) < lengths.unsqueeze(-1))
        
        # Flat list of all valid tokens
        # Note: This flattens the batch structure too, we need to be careful if Batch > 1
        # For RL rollouts (Batch=1), return compact sequence
        if B == 1:
            seq = fragments[mask].unsqueeze(0)
            if return_attention_mask:
                attn = torch.ones_like(seq, dtype=torch.long, device=self.device)
                return seq, attn
            return seq
        
        # For B > 1, we must handle ragged tensors (complex). 
        # Since your rollout loop is likely B=1 or we can pad:
        # We will flatten the last two dims and just let PADs exist (Guidance model ignores them via attention mask)
        # But for simplicity, let's just flatten:
        flat_fragments = fragments.view(B, -1)
        # (This is a naive flatten that includes garbage. 
        #  Proper ragged batching requires specialized kernels, 
        #  but usually Guidance handles slight noise or we just mask it.)
        
        # Better approximation for Training: Just take the FIRST fragment if multiple exist
        # This keeps shapes consistent [B, L] -> [B, L]
        # It loses some info (e.g. "banana" split), but assumes Guidance is robust.
        # IF YOU NEED PERFECT ACCURACY, USE THE MASK METHOD ABOVE (works great for Batch=1).
        
        # For B > 1, pad to max length in batch
        flat_list = []
        max_len = 0
        for b in range(B):
            seq_b = fragments[b][mask[b]]
            flat_list.append(seq_b)
            max_len = max(max_len, seq_b.numel())

        padded = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
        attn = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
        for b in range(B):
            seq_b = flat_list[b]
            if seq_b.numel() > 0:
                padded[b, :seq_b.numel()] = seq_b
                attn[b, :seq_b.numel()] = 1

        return (padded, attn) if return_attention_mask else padded

    def align_logits(self, src_logits: torch.Tensor, tgt_vocab_size: int, topk: int = 50) -> torch.Tensor:
        """
        Identical to RobustAligner, but fully cached.
        """
        probs = torch.softmax(src_logits, dim=-1)
        topk_probs, topk_src_ids = probs.topk(topk, dim=-1)

        batch_size = src_logits.size(0)
        tgt_logits = torch.full((batch_size, tgt_vocab_size), float('-inf'), device=self.device, dtype=src_logits.dtype)

        max_src_id = self.logit_map_ids.size(0)
        valid_src_mask = (topk_src_ids >= 0) & (topk_src_ids < max_src_id)
        safe_src_ids = torch.where(valid_src_mask, topk_src_ids, torch.tensor(0, device=self.device))

        mapped_ids = self.logit_map_ids[safe_src_ids]  # [B, K, M]
        mapped_lens = self.logit_map_lens[safe_src_ids]  # [B, K]
        M = mapped_ids.size(-1)
        token_mask = torch.arange(M, device=self.device).view(1, 1, M) < mapped_lens.unsqueeze(-1)
        token_mask = token_mask & valid_src_mask.unsqueeze(-1)
        token_mask = token_mask & (mapped_ids < tgt_vocab_size)

        log_vals = torch.log(topk_probs + 1e-10).unsqueeze(-1).expand_as(mapped_ids)

        if batch_size == 1:
            flat_ids = mapped_ids[0][token_mask[0]]
            flat_vals = log_vals[0][token_mask[0]]
            if flat_ids.numel() > 0:
                tgt_logits[0].scatter_reduce_(0, flat_ids, flat_vals, reduce='max', include_self=False)
        else:
            for b in range(batch_size):
                ids_b = mapped_ids[b][token_mask[b]]
                vals_b = log_vals[b][token_mask[b]]
                if ids_b.numel() > 0:
                    tgt_logits[b].scatter_reduce_(0, ids_b, vals_b, reduce='max', include_self=False)

        return tgt_logits

    def align(self, src_logits: torch.Tensor, tgt_vocab_size: int, topk: int = 50) -> torch.Tensor:
        return self.align_logits(src_logits, tgt_vocab_size, topk=topk)
    
