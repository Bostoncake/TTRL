# Multi-Modal TTRL for SAT2 Spatial Reasoning

This directory contains the implementation for adapting TTRL (Test-Time Reinforcement Learning) to work with **Multi-Modal Large Language Models** (Qwen-2.5-VL-3B) on the **SAT2 spatial reasoning task**.

## Quick Start

### 1. Install Dependencies
```bash
cd verl
pip install -e .
pip install qwen-vl-utils Pillow datasets
```

### 2. Preprocess SAT2 Data
```bash
python verl/examples/data_preprocess/sat2.py \
  --input_file sat2_test.jsonl \
  --local_dir ~/data/SAT2-SPATIAL \
  --base_image_dir .
```

### 3. Download Model
```bash
python3 -c "from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor; \
  Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct'); \
  Qwen2VLProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')"
```

### 4. Run TTRL Training
```bash
export BACKBONE_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
export DATA_LOCAL_DIR="$HOME/data"
bash verl/examples/ttrl/Qwen2.5-VL/sat2.sh
```

## File Structure

```
TTRL/
├── TTRL_MULTIMODAL_SAT2_GUIDE.md          # Comprehensive implementation guide
├── sat2_test.jsonl                         # SAT2 dataset (provided)
├── verl/
│   ├── examples/
│   │   ├── data_preprocess/
│   │   │   └── sat2.py                     # NEW: SAT2 data preprocessing
│   │   └── ttrl/
│   │       └── Qwen2.5-VL/
│   │           └── sat2.sh                 # NEW: TTRL training script
│   └── verl/
│       └── utils/
│           └── reward_score/
│               └── sat2/
│                   └── __init__.py         # NEW: SAT2 reward function
```

## Key Modifications

### 1. Data Preprocessing (`verl/examples/data_preprocess/sat2.py`)
- Converts SAT2 JSONL to VERL parquet format
- Handles image paths and multi-turn conversations
- Creates train/test split

### 2. Reward Function (`verl/verl/utils/reward_score/sat2/__init__.py`)
- Text-matching reward for spatial reasoning answers
- Extracts answers from model responses
- Fuzzy matching with normalization

### 3. Training Script (`verl/examples/ttrl/Qwen2.5-VL/sat2.sh`)
- Configures Qwen2.5-VL-3B for TTRL
- Sets hyperparameters for multi-modal training
- Enables majority voting with 32 samples

## What Makes This Different?

### From Text-Only TTRL (Math)
- **Input**: Text + Images (instead of text only)
- **Model**: Qwen2.5-VL-3B (vision-language) instead of Qwen2.5-Math-7B
- **Answer Format**: Multiple-choice text instead of boxed math answers
- **Reward**: Simple text matching instead of symbolic math checking

### Implementation Simplicity
- ✅ **0 lines changed** in core TTRL code (`ttrl_utils.py`, `ray_trainer.py`)
- ✅ **~400 lines** of task-specific code (data, reward, config)
- ✅ **Existing multi-modal support** in VERL handles image processing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Input: <image> + "Did the camera move?"                    │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Qwen2.5-VL-3B: Process image + text → Generate responses   │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ TTRL Majority Voting: 32 responses → Pseudo-label          │
│ "did not move" (20 votes), "rotated left" (12 votes)       │
│ → Majority: "did not move"                                  │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Reward: Compare 16 sampled responses to majority           │
│ Score: 1.0 if match, 0.0 otherwise                         │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ PPO Update: Increase probability of majority-aligned       │
│ responses using GRPO advantages                             │
└─────────────────────────────────────────────────────────────┘
```

## Expected Results

| Metric | Before TTRL | After TTRL (50 epochs) |
|--------|-------------|------------------------|
| Greedy Accuracy | ~45-50% | ~60-65% (+15%) |
| Pass@16 | ~55-60% | ~70-75% (+15%) |
| Majority@16 | ~52-58% | Surpassed by greedy ✅ |

## Documentation

- **Comprehensive Guide**: See `TTRL_MULTIMODAL_SAT2_GUIDE.md` for detailed implementation walkthrough
- **Original TTRL Guide**: See `TTRL_IMPLEMENTATION_GUIDE.md` for text-only TTRL explanation

## Troubleshooting

### Memory Issues
```bash
# Reduce batch size
data.train_batch_size=2

# Reduce TTRL samples
ttrl.n_votes_per_prompt=16
ttrl.n_samples_per_prompt=8
```

### Image Loading Errors
```bash
# Check image paths in preprocessing
python -c "
import pandas as pd
df = pd.read_parquet('~/data/SAT2-SPATIAL/train.parquet')
print(df['images'][0])  # Should print full path
import os
print(os.path.exists(df['images'][0][0]))  # Should be True
"
```

## Key Insights

1. **TTRL is Task-Agnostic**: The core majority voting logic works for any task (math, code, spatial reasoning)

2. **Multi-Modal Support is Built-In**: VERL handles images automatically via `data.image_key=images`

3. **Simple Reward Functions Work**: You don't need complex grading—text matching is sufficient for multiple-choice tasks

4. **Memory is the Main Challenge**: Vision-language models require more GPU memory than text-only models

## Citation

If you use this implementation, please cite:

```bibtex
@article{ttrl2024,
  title={TTRL: Test-Time Reinforcement Learning},
  author={...},
  journal={arXiv preprint arXiv:2504.16084},
  year={2024}
}
```

## Contact

For questions or issues:
- See `TTRL_MULTIMODAL_SAT2_GUIDE.md` for detailed troubleshooting
- Open a GitHub issue in the TTRL repository
