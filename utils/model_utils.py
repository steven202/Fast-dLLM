def get_mask_id(model_type: str, model) -> int:
    if hasattr(model, "config") and hasattr(model.config, "mask_token_id") and model.config.mask_token_id is not None:
        return model.config.mask_token_id
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "mask_token_id") and model.generation_config.mask_token_id is not None:
        return model.generation_config.mask_token_id
    if model_type == "llada":
        return 126336
    raise ValueError("Unable to infer mask token id.")
