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

    def translate_input(self, tgt_ids: torch.Tensor) -> torch.Tensor:
        """
        Maps Backbone IDs (Tgt) -> Guidance IDs (Src) for input.
        """
        max_id = self.input_map.size(0)
        valid_mask = (tgt_ids < max_id)
        safe_ids = torch.where(valid_mask, tgt_ids, torch.tensor(0, device=self.device))
        
        mapped_ids = self.input_map[safe_ids]
        
        # If no mapping (-1) or invalid input, use 0 (UNK/PAD assumption)
        out_ids = torch.where((mapped_ids != -1) & valid_mask, mapped_ids, torch.tensor(0, device=self.device))
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
