# Fast-dLLM Baseline Results - 2026-04-15

## Configuration
- Model: Dream-org/Dream-v0-Base-7B
- max_new_tokens: 256
- block_length: 32
- alg: confidence_threshold
- threshold: 0.9
- use_cache: true
- dual_cache: true
- escape_until: true (HumanEval only)

## Results

### s=8

| Dataset | Quality | TPS | NFE | TPF |
|---------|---------|-----|-----|-----|
| GSM8K | 0.74 | 59.82 | 16345 | 1.56 |
| HumanEval | 0.63 | 69.62 | 15247 | 1.67 |

### s=256

| Dataset | Quality | TPS | NFE | TPF |
|---------|---------|-----|-----|-----|
| GSM8K | 0.74 | 59.78 | 16345 | 1.56 |
| HumanEval | 0.63 | 34.87 | 15247 | 1.67 |

## Notes

- HumanEval pass@1 calculated using postprocess_code.py (lm_eval reports wrong value ~0.03)
- GSM8K quality from exact_match,flexible-extract
- NFE and TPF metrics added to Fast-dLLM eval.py for this run
