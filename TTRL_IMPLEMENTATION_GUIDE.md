# TTRL Implementation Walkthrough Guide

## Table of Contents
1. [Overview](#overview)
2. [TTRL Methodology Summary](#ttrl-methodology-summary)
3. [Code Architecture](#code-architecture)
4. [Data Preparation Pipeline](#data-preparation-pipeline)
5. [Configuration Setup](#configuration-setup)
6. [Core TTRL Components](#core-ttrl-components)
7. [Training Pipeline](#training-pipeline)
8. [Detailed Implementation Walkthrough](#detailed-implementation-walkthrough)
9. [Complete Workflow Example](#complete-workflow-example)

---

## Overview

This guide provides a detailed walkthrough of how the TTRL (Test-Time Reinforcement Learning) research paper (arXiv:2504.16084) is implemented in this codebase. TTRL enables self-evolution of LLMs using RL on unlabeled test data by using majority voting as a reward signal.

**Key Innovation**: TTRL uses majority voting from multiple model outputs to create pseudo-labels for test data, then uses these pseudo-labels to drive RL training, achieving performance that surpasses the initial model's maj@n baseline.

---

## TTRL Methodology Summary

### Core Idea
- **Problem**: Train LLMs on data without ground-truth labels (especially test data)
- **Solution**: Use majority voting from multiple samples to estimate rewards
- **Result**: Models improve beyond their initial maj@n performance ceiling

### Key Components
1. **Majority Voting**: Generate N samples per prompt and use the most common answer as pseudo-ground-truth
2. **Downsampling**: Generate more samples for voting (e.g., 64) than used for training (e.g., 32) to improve label quality
3. **Reward Function**: Compare model outputs against majority-voted labels
4. **PPO Training**: Use GRPO (Group Relative Policy Optimization) as the advantage estimator

---

## Code Architecture

### Directory Structure
```
verl/
├── data/
│   ├── preprocess.py                    # Data preprocessing
│   └── [dataset folders]/               # Train/test data in JSON/Parquet
├── verl/
│   ├── trainer/
│   │   ├── ppo/
│   │   │   ├── main_ppo.py             # Main entry point
│   │   │   ├── ray_trainer.py          # Core PPO trainer with TTRL integration
│   │   │   ├── ttrl_utils.py           # TTRL-specific utilities
│   │   │   ├── reward.py               # Reward manager
│   │   │   └── core_algos.py           # PPO algorithms (GAE, GRPO, etc.)
│   │   └── config/
│   │       └── ppo_trainer_ttrl.yaml   # TTRL configuration
│   └── utils/
│       └── reward_score/
│           └── ttrl_math/
│               └── __init__.py          # Math reward function
└── examples/
    └── ttrl/
        └── Qwen2.5/
            └── aime.sh                  # Training script
```

---

## Data Preparation Pipeline

### File: `/verl/data/preprocess.py`

**Purpose**: Convert JSON data to Parquet format for VERL training

#### Key Function: `make_map_fn` (lines 6-33)

```python
def make_map_fn(split, source=None):
    def process_fn(example, idx):
        # Extract fields from raw data
        data_source = example.pop("source")  # Line 9
        question = example.pop("prompt")      # Line 12
        solution = example.pop("answer")      # Line 13

        # Create structured data format
        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],  # Line 18-22
            "ability": "math",                                   # Line 24
            "reward_model": {
                "style": "rule",
                "ground_truth": solution  # Original ground truth stored here
            },                                                   # Line 25
            "extra_info": {
                "split": split,
                "index": f"{data_source}-{idx}"                 # Line 27-29
            }
        }
        return data
```

**What It Does**:
1. Line 9: Extracts the data source identifier
2. Line 12-13: Extracts prompt and answer from raw data
3. Line 18-22: Formats prompt as a chat message (user role)
4. Line 25: **Stores original ground truth** in `reward_model.ground_truth` - this is crucial for TTRL
5. Line 28: Creates unique index for each example

**Data Format**:
- Input: JSON with `prompt`, `answer`, `source` fields
- Output: Parquet with structured format including ground truth

**Example Usage** (lines 35-46):
```bash
python verl/data/preprocess.py
# Converts train.json → train.parquet
# Converts test.json → test.parquet
```

---

## Configuration Setup

### File: `/verl/verl/trainer/config/ppo_trainer_ttrl.yaml`

**Purpose**: TTRL-specific configuration extending base PPO trainer

#### TTRL Configuration Block (lines 9-17)

```yaml
ttrl:
  enable: false                          # Line 11: Master switch for TTRL
  n_samples_per_prompt: ${actor_rollout_ref.rollout.n}  # Line 14: Samples for training
  n_votes_per_prompt: ${actor_rollout_ref.rollout.n}    # Line 17: Samples for voting
```

**Parameters Explained**:
- **`enable`** (line 11): Toggle TTRL on/off
- **`n_samples_per_prompt`** (line 14): Number of rollouts used for actual training (e.g., 32)
- **`n_votes_per_prompt`** (line 17): Number of rollouts for majority voting (e.g., 64)
  - Can be larger than `n_samples_per_prompt` for better label quality
  - Implements the "downsampling" strategy from the paper

**Usage in Script**: See `/verl/examples/ttrl/Qwen2.5/aime.sh` lines 91-93

```bash
ttrl.enable=True \
ttrl.n_votes_per_prompt=$N_VOTES_PER_PROMPT \    # Line 92: Set to 64
ttrl.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \ # Line 93: Set to 32
```

---

## Core TTRL Components

### File: `/verl/verl/trainer/ppo/ttrl_utils.py`

This file contains the heart of TTRL's majority voting implementation.

---

#### 1. Majority Voting Function

**Function**: `_majority_vote` (lines 109-122)

```python
def _majority_vote(model_outputs: List[str]) -> tuple[str, float]:
    """Generate pseudo-ground-truth via majority voting"""

    # Extract answers from model outputs
    model_answers = [extract_answer(generated_text) for generated_text in model_outputs]  # Line 111

    # Filter out None answers
    model_answers = [answer for answer in model_answers if answer is not None]  # Line 112

    # Simplify expressions for comparison (e.g., "1/2" == "0.5")
    model_answers = [simplify_expression_string(answer) for answer in model_answers]  # Line 113

    # If no valid answers, return None
    if len(model_answers) == 0:
        return "None", 0.0  # Line 115

    # Count occurrences of each answer
    counter = Counter(model_answers)  # Line 117

    # Get most common answer
    majority_answer, majority_count = counter.most_common(1)[0]  # Line 119

    # Calculate confidence (majority ratio)
    majority_ratio = majority_count / len(model_outputs)  # Line 120

    return majority_answer, majority_ratio  # Line 122
```

**What It Does**:
1. **Line 111**: Extracts answers from raw model outputs (e.g., from `\boxed{...}` in LaTeX)
2. **Line 112**: Filters invalid/unparseable answers
3. **Line 113**: Normalizes answers (e.g., simplifies mathematical expressions)
4. **Line 117**: Uses `Counter` to count each unique answer
5. **Line 119**: Selects the most frequent answer as pseudo-label
6. **Line 120**: Calculates confidence as the fraction agreeing with majority

**Key Insight**: This implements the core TTRL idea - using model consensus as a supervision signal.

---

#### 2. Batch Majority Voting

**Function**: `_batch_majority_vote` (lines 86-106)

```python
def _batch_majority_vote(model_outputs: List[str], n: int) -> tuple[List[str], List[float]]:
    """Process multiple prompts in batch"""

    majority_gt_list = []
    majority_ratio_list = []

    assert len(model_outputs) % n == 0  # Line 98: Ensure divisibility
    n_prompts = len(model_outputs) // n  # Line 99: Calculate number of prompts

    # Process each prompt separately
    for i in range(n_prompts):  # Line 100
        # Extract n outputs for this prompt
        prompt_outputs = model_outputs[i * n:(i + 1) * n]  # Line 101

        # Compute majority vote for this prompt
        prompt_majority_gt, prompt_majority_ratio = _majority_vote(prompt_outputs)  # Line 102

        majority_gt_list.append(prompt_majority_gt)      # Line 103
        majority_ratio_list.append(prompt_majority_ratio)  # Line 104

    return majority_gt_list, majority_ratio_list  # Line 106
```

**What It Does**:
1. **Line 98-99**: Validates and calculates batch structure
2. **Line 101**: Slices outputs for each prompt (groups of n)
3. **Line 102**: Applies majority voting to each group
4. **Line 103-104**: Collects results

---

#### 3. Apply TTRL Ground Truth

**Function**: `apply_ttrl_gt` (lines 50-83)

This is where TTRL replaces original ground truth with majority-voted labels.

```python
def apply_ttrl_gt(batch, gen_batch_output, n, tokenizer):
    """Replace ground truth with majority-voted labels"""

    assert len(gen_batch_output) % n == 0  # Line 54
    num_prompts = len(gen_batch_output) // n  # Line 55
    assert len(batch) == num_prompts  # Line 56

    # Decode model outputs
    model_outputs = []
    for i in range(num_prompts):  # Line 59
        start = i * n
        for j in range(n):  # Line 61
            data_item = gen_batch_output[start + j]

            # Extract prompt and response
            prompt_ids = data_item.batch["prompts"]  # Line 63
            prompt_length = prompt_ids.shape[-1]  # Line 64
            response_ids = data_item.batch["responses"]  # Line 65

            # Get valid response length
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()  # Line 66
            valid_response_ids = response_ids[:valid_response_length]  # Line 67

            # Decode to string
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)  # Line 68
            model_outputs.append(response_str)  # Line 69

    # Compute majority vote for all prompts
    majority_gt_list, majority_ratio_list = _batch_majority_vote(model_outputs, n)  # Line 71

    # Update batch with majority-voted ground truth
    for i in range(num_prompts):  # Line 75
        data_item = batch[i]

        # Save original ground truth
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]  # Line 77

        # Replace with majority-voted label
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]  # Line 78
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]  # Line 79
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt  # Line 80

    # Store majority ratios for metrics
    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)  # Line 82
    return batch  # Line 83
```

**Critical Steps**:
1. **Lines 59-69**: Decode all generated responses to text
2. **Line 71**: Compute majority vote for each prompt
3. **Line 77**: **Save original ground truth** (important for metrics)
4. **Line 78**: **Replace ground truth with majority vote** - this is the TTRL magic!
5. **Line 79**: Store majority label separately for tracking
6. **Line 82**: Store confidence scores

---

#### 4. Downsampling Function

**Function**: `select_top_k_per_prompt` (lines 20-32)

```python
def select_top_k_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt):
    """Select first k rollouts per prompt for training"""

    assert len(data) % n_votes_per_prompt == 0  # Line 24: Validate structure
    num_prompts = len(data) // n_votes_per_prompt  # Line 25

    selected_indices = []
    for i in range(num_prompts):  # Line 28
        start = i * n_votes_per_prompt
        # Select first n_samples_per_prompt from each group
        selected_indices.extend(range(start, start + n_samples_per_prompt))  # Line 30

    return data[selected_indices]  # Line 32
```

**What It Does**:
- **Input**: 64 samples per prompt (for voting)
- **Output**: 32 samples per prompt (for training)
- **Purpose**: Use all 64 for better voting, but train on only 32 to save computation

**Example**:
- Generate 64 samples → Use all 64 for voting → Train on first 32
- This improves label quality while controlling training cost

---

#### 5. TTRL Metrics Computation

**Function**: `compute_ttrl_metrics` (lines 128-155)

```python
def compute_ttrl_metrics(batch, n):
    """Compute metrics comparing TTRL vs ground truth"""

    # Sort by index for consistency
    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])  # Line 136

    majority_reward = []
    gt_reward = []
    majority_label = []
    gt_label = []

    # Collect rewards and labels
    for i in range(len(batch)):  # Line 143
        data_item = batch[idx[i]]
        majority_reward.append(data_item.batch["token_level_scores"].sum().item())  # Line 145
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())  # Line 146
        majority_label.append(data_item.non_tensor_batch["reward_model"]["majority_gt"])  # Line 147
        gt_label.append(data_item.non_tensor_batch["reward_model"]["original_gt"])  # Line 148

    # Compute batch metrics
    ttrl_metrics = _batch_compute_ttrl_metrics(
        majority_reward, gt_reward, majority_label, gt_label, n=n
    )  # Line 150

    # Add majority ratio
    majority_ratio_list = batch.non_tensor_batch["majority_ratio_list"]  # Line 151
    majority_ratio = sum(majority_ratio_list) / len(majority_ratio_list)  # Line 152
    ttrl_metrics["majority_ratio"] = majority_ratio  # Line 153

    return ttrl_metrics  # Line 155
```

**Metrics Computed** (see `_prompt_compute_ttrl_metrics`, lines 192-214):
- **`label_accuracy`** (line 208): Does majority vote match true answer?
- **`reward_accuracy`** (line 209): Do majority-based and true rewards match?
- **`majority_voting_reward`** (line 210): Average reward from majority labels
- **`ground_truth_reward`** (line 211): Average reward from true labels
- **`pass@n`** (line 212): Success rate across n samples
- **`majority_ratio`** (line 153): Average confidence of majority votes

---

## Training Pipeline

### File: `/verl/verl/trainer/ppo/main_ppo.py`

This is the entry point for TTRL training.

#### Main Function (lines 29-31)

```python
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)  # Line 31
```

#### PPO Runner (lines 34-214)

Key steps in the training pipeline:

1. **Ray Initialization** (lines 42-52)
2. **Tokenizer Loading** (lines 96-101)
3. **Worker Class Selection** (lines 112-137)
4. **Reward Manager Setup** (lines 180-185)

```python
# Load reward manager for training and validation
reward_fn = load_reward_manager(
    config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
)  # Line 180-182
val_reward_fn = load_reward_manager(
    config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
)  # Line 183-185
```

5. **Dataset Creation** (lines 191-193)
6. **Trainer Initialization** (lines 196-210)
7. **Training Start** (lines 212-214)

---

### File: `/verl/verl/trainer/ppo/ray_trainer.py`

This file contains the core training loop with TTRL integration.

---

#### TTRL Integration Point 1: Generation Phase

**Location**: Lines 1137-1148

```python
# Check if TTRL is enabled
if self.config.get("ttrl", {}).get("enable", False):  # Line 1137
    from verl.trainer.ppo.ttrl_utils import select_top_k_per_prompt, apply_ttrl_gt  # Line 1138

    # Set number of samples for voting
    gen_batch.meta_info["kwargs"] = {"n": self.config.ttrl.n_votes_per_prompt}  # Line 1140

    # Generate n_votes_per_prompt samples (e.g., 64)
    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)  # Line 1141

    # Verify we got the right number
    assert len(gen_batch_output) == len(batch) * self.config.ttrl.n_votes_per_prompt  # Line 1143

    # Apply majority voting to create pseudo-labels
    batch = apply_ttrl_gt(batch, gen_batch_output, self.config.ttrl.n_votes_per_prompt, self.tokenizer)  # Line 1145

    # Downsample to n_samples_per_prompt (e.g., 32)
    gen_batch_output = select_top_k_per_prompt(
        gen_batch_output,
        self.config.ttrl.n_votes_per_prompt,
        self.config.ttrl.n_samples_per_prompt
    )  # Line 1146

    # Verify downsampling
    assert len(gen_batch_output) == len(batch) * self.config.ttrl.n_samples_per_prompt  # Line 1148
else:
    # Standard PPO generation
    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)  # Line 1151
```

**What Happens Here**:
1. **Line 1140**: Configure rollout to generate 64 samples per prompt
2. **Line 1141**: Generate all samples
3. **Line 1145**: **Core TTRL step** - replace ground truth with majority vote
4. **Line 1146**: Downsample from 64 to 32 samples for training
5. **Line 1148**: Validate final batch size

---

#### TTRL Integration Point 2: Metrics Computation

**Location**: Lines 1311-1319

```python
# After actor update, compute TTRL metrics
if self.config.get("ttrl", {}).get("enable", False):  # Line 1311
    from verl.trainer.ppo.ttrl_utils import apply_original_gt, compute_ttrl_metrics  # Line 1312

    # Restore original ground truth
    batch = apply_original_gt(batch)  # Line 1313

    # Compute reward with original labels
    reward_tensor_original, reward_extra_infos_dict_original = compute_reward(batch, self.reward_fn)  # Line 1314
    batch.batch["token_level_scores_original"] = reward_tensor_original  # Line 1315

    # Compute TTRL-specific metrics
    ttrl_metrics = compute_ttrl_metrics(batch, self.config.ttrl.n_samples_per_prompt)  # Line 1317

    # Log metrics
    for key, value in ttrl_metrics.items():  # Line 1318
        metrics.update({f"train/{key}": value})  # Line 1319
```

**What Happens Here**:
1. **Line 1313**: Restore original ground truth (was replaced in line 1145)
2. **Line 1314-1315**: Compute reward with **true** labels for comparison
3. **Line 1317**: Calculate TTRL metrics (accuracy, majority ratio, etc.)
4. **Line 1319**: Log metrics to W&B/console

---

## Detailed Implementation Walkthrough

### Complete TTRL Training Step

Let's trace what happens when TTRL processes one batch:

---

#### Step 1: Data Loading

**File**: `/verl/verl/trainer/main_ppo.py`, lines 191-193

```python
train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
```

- Loads parquet data with `reward_model.ground_truth` field
- Each example has original ground truth stored

---

#### Step 2: Batch Preparation

**File**: `/verl/verl/trainer/ppo/ray_trainer.py`, lines 1101-1120

```python
batch: DataProto = DataProto.from_single_dict(batch_dict)  # Line 1101
gen_batch = batch.pop(batch_keys=batch_keys_to_pop, ...)  # Line 1117
```

- Separates prompt data (`gen_batch`) from full batch
- `batch` contains ground truth labels

---

#### Step 3: Generate N Samples (TTRL Mode)

**File**: `/verl/verl/trainer/ppo/ray_trainer.py`, line 1140-1141

```python
gen_batch.meta_info["kwargs"] = {"n": self.config.ttrl.n_votes_per_prompt}  # 64 samples
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
```

- Generates 64 responses per prompt
- Uses vLLM for efficient parallel generation

---

#### Step 4: Majority Voting

**File**: `/verl/verl/trainer/ppo/ray_trainer.py`, line 1145
**Calls**: `/verl/verl/trainer/ppo/ttrl_utils.py`, line 50

```python
batch = apply_ttrl_gt(batch, gen_batch_output, n_votes_per_prompt=64, tokenizer)
```

Inside `apply_ttrl_gt`:
1. Decode all 64 responses to text (lines 63-69)
2. Call `_batch_majority_vote(model_outputs, n=64)` (line 71)
   - For each prompt: extract answers, count occurrences, select majority
3. **Replace `batch.reward_model.ground_truth`** with majority answer (line 78)
4. Save original ground truth to `original_gt` (line 80)

**Result**: Batch now has pseudo-labels from majority voting

---

#### Step 5: Downsampling

**File**: `/verl/verl/trainer/ppo/ray_trainer.py`, line 1146
**Calls**: `/verl/verl/trainer/ppo/ttrl_utils.py`, line 20

```python
gen_batch_output = select_top_k_per_prompt(
    gen_batch_output, n_votes_per_prompt=64, n_samples_per_prompt=32
)
```

- Keeps only first 32 out of 64 samples per prompt
- Reduces training cost while maintaining label quality

---

#### Step 6: Union Batch

**File**: `/verl/verl/trainer/ppo/ray_trainer.py`, line 1182

```python
batch = batch.union(gen_batch_output)
```

- Combines prompt data with generated responses
- Batch now has: prompts, responses, pseudo-labels, original labels

---

#### Step 7: Reward Computation

**File**: `/verl/verl/trainer/ppo/ray_trainer.py`, around line 1197-1220

```python
# Compute reward using MAJORITY-VOTED labels
reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
batch.batch["token_level_scores"] = reward_tensor
```

**Calls**: `/verl/verl/utils/reward_score/ttrl_math/__init__.py`, line 105

```python
def reward_func(data_source, solution_str, ground_truth, ...):
    res = compute_score(solution_str, str(ground_truth))  # Line 109
    # Returns 1.0 if correct, 0.0 if wrong
```

**Critical**: `ground_truth` here is the **majority-voted answer**, not the original!

---

#### Step 8: Advantage Calculation (GRPO)

**File**: `/verl/verl/trainer/ppo/ray_trainer.py`, lines 1284-1293

```python
batch = compute_advantage(
    batch,
    adv_estimator=AdvantageEstimator.GRPO,  # Group Relative Policy Optimization
    ...
)
```

**Calls**: `/verl/verl/trainer/ppo/core_algos.py`, line 205

```python
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, ...):
    scores = token_level_rewards.sum(dim=-1)  # Line 232: Sum token rewards to outcome reward

    # Group by prompt ID
    for i in range(bsz):
        id2score[index[i]].append(scores[i])  # Line 241

    # Compute mean and std per group
    for idx in id2score:
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))  # Line 247
        id2std[idx] = torch.std(torch.tensor([id2score[idx]]))  # Line 248

    # Normalize advantages
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)  # Line 253
```

**GRPO Explanation**:
- Groups responses by prompt (using `index`)
- Normalizes each response's reward by group mean/std
- Responses above group average get positive advantage
- Responses below group average get negative advantage

---

#### Step 9: Actor Update

**File**: `/verl/verl/trainer/ppo/ray_trainer.py`, lines 1305-1309

```python
# Update policy using PPO
actor_output = self.actor_rollout_wg.update_actor(batch)
```

- Standard PPO update using computed advantages
- Policy learns to produce responses that:
  - Match majority-voted answers
  - Have higher outcome rewards than group average

---

#### Step 10: Metrics Computation

**File**: `/verl/verl/trainer/ppo/ray_trainer.py`, lines 1311-1319

```python
if self.config.get("ttrl", {}).get("enable", False):
    # Restore original labels
    batch = apply_original_gt(batch)  # Line 1313

    # Compute reward with TRUE labels
    reward_tensor_original, _ = compute_reward(batch, self.reward_fn)  # Line 1314
    batch.batch["token_level_scores_original"] = reward_tensor_original  # Line 1315

    # Compute metrics comparing TTRL vs ground truth
    ttrl_metrics = compute_ttrl_metrics(batch, n_samples_per_prompt=32)  # Line 1317
    metrics.update({f"train/{key}": value for key, value in ttrl_metrics.items()})  # Line 1319
```

**Metrics Logged**:
- `train/label_accuracy`: % of times majority vote = true answer
- `train/reward_accuracy`: % of times rewards match
- `train/majority_voting_reward`: Reward from TTRL labels
- `train/ground_truth_reward`: Reward from true labels
- `train/pass@32`: Success rate
- `train/majority_ratio`: Average voting confidence

---

## Complete Workflow Example

Let's trace a concrete example from the AIME training script.

### File: `/verl/examples/ttrl/Qwen2.5/aime.sh`

---

#### Configuration (lines 11-29)

```bash
TASK="AIME-TTT"
BACKBONE="Qwen2.5-7B"
ADVANTAGE="grpo"                 # Use GRPO advantage estimator

K=3                               # Response length multiplier
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=$((1024 * $K))  # 3072 tokens

N=16                              # Line 21: Validation samples
EPISODE=80                        # Line 24: Training epochs
DATA_TRAIN_BATCH_SIZE=8          # Line 25: Prompts per batch

N_VOTES_PER_PROMPT=64            # Line 26: Samples for voting
N_SAMPLES_PER_PROMPT=32          # Line 27: Samples for training

MINI_BATCH_SIZE=1                # Line 28: PPO mini-batch size
MICRO_BATCH_SIZE=2               # Line 29: Micro-batch size
```

**Key Parameters**:
- 64 samples for voting, 32 for training (2:1 ratio)
- GRPO advantage estimator
- 8 prompts per batch
- 80 training epochs

---

#### Command Execution (lines 42-104)

```bash
python -m verl.trainer.main_ppo \
  --config-name='ppo_trainer_ttrl.yaml' \  # Line 43: Use TTRL config

  # Data configuration
  data.train_files=["$DATA_LOCAL_DIR/$TASK/train.parquet"] \  # Line 44
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \

  # Prompt suffix
  +data.suffix_prompt='"\nPlease reason step by step, and put your final answer within \boxed{}."' \  # Line 51

  # Model configuration
  actor_rollout_ref.model.path=$BACKBONE_PATH \  # Line 52

  # Rollout configuration
  actor_rollout_ref.rollout.name=vllm \  # Line 66: Use vLLM for generation
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \  # Line 73: 32 samples

  # Algorithm configuration
  algorithm.kl_ctrl.kl_coef=0.00 \  # Line 87: No KL penalty
  algorithm.adv_estimator=$ADVANTAGE \  # Line 88: GRPO

  # Custom reward function
  custom_reward_function.path="./verl/utils/reward_score/ttrl_math/__init__.py" \  # Line 89
  custom_reward_function.name=reward_func \  # Line 90

  # TTRL configuration
  ttrl.enable=True \  # Line 91: Enable TTRL
  ttrl.n_votes_per_prompt=$N_VOTES_PER_PROMPT \  # Line 92: 64 for voting
  ttrl.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \  # Line 93: 32 for training

  # Training configuration
  trainer.total_epochs=$EPISODE  # Line 104: 80 epochs
```

---

#### Execution Flow

1. **Initialization**:
   - Load Qwen2.5-7B model
   - Initialize vLLM engine
   - Load AIME training data

2. **For each of 80 epochs**:
   - **For each batch of 8 prompts**:

     a. **Generation** (64 samples per prompt = 512 total):
        - Append suffix: "Please reason step by step, and put your final answer within \boxed{}."
        - Generate 64 responses per prompt with temperature=0.6

     b. **Majority Voting**:
        - For each of 8 prompts:
          - Extract answers from 64 responses
          - Count answer frequencies
          - Select most common answer as pseudo-label
          - Calculate confidence (majority ratio)

     c. **Downsampling**:
        - Keep first 32 of 64 samples per prompt
        - Results in 256 training samples (8 prompts × 32 samples)

     d. **Reward Computation**:
        - Compare each of 256 responses to its prompt's majority-voted answer
        - Score: 1.0 if match, 0.0 if mismatch

     e. **GRPO Advantage**:
        - Group 32 responses by prompt
        - Compute mean reward per group
        - Normalize: advantage = (reward - group_mean) / group_std

     f. **PPO Update**:
        - Update policy to increase probability of high-advantage responses
        - Clip policy updates to prevent large changes

     g. **Metrics**:
        - Restore original ground truth
        - Compute reward with true labels
        - Log comparison metrics:
          - How often does majority vote match true answer?
          - How much reward improvement?
          - What's the average majority confidence?

3. **Validation** (every 2 epochs):
   - Generate 16 samples per validation prompt
   - Compute pass@16 and other metrics

---

## Key Implementation Details

### 1. Math Answer Extraction

**File**: `/verl/verl/utils/reward_score/ttrl_math/__init__.py`

```python
def extract_answer(passage: str) -> str:  # Line 30
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)  # Line 32: Extract from \boxed{...}
    return None  # Line 33
```

- Looks for LaTeX `\boxed{answer}` format
- Returns None if no boxed answer found

---

### 2. Answer Simplification

**File**: `/verl/verl/utils/reward_score/ttrl_math/__init__.py`

```python
@timeout_ours(timeout_seconds=10)  # Line 49: Prevent infinite loops
def simplify_expression_string(expression_string: str) -> str:  # Line 50
    try:
        # Try parsing as SymPy expression
        sympy_expr = parse_expr(expression_string, transformations="all", evaluate=False)  # Line 52
        simplified_expr = simplify(sympy_expr)  # Line 53
        return str(simplified_expr)  # Line 54
    except Exception:
        try:
            # Try LaTeX to SymPy conversion
            sympy_expr = latex2sympy(expression_string)  # Line 59
            simplified_expr = simplify(sympy_expr)  # Line 60
            return str(simplified_expr)  # Line 61
        except Exception:
            # Return original if all fails
            return expression_string  # Line 65
```

**Purpose**: Normalize different representations of same answer
- `"1/2"`, `"0.5"`, `"\frac{1}{2}"` → all become same simplified form

---

### 3. Answer Grading

**File**: `/verl/verl/utils/reward_score/ttrl_math/__init__.py`

```python
def grade(model_answer: str, gt_answer: str, fast: bool = True):  # Line 36
    if "\\boxed" in gt_answer:
        gt_answer = extract_answer(gt_answer)  # Line 38: Extract if needed

    # Try multiple grading methods
    correct = grade_answer_mathd(model_answer, gt_answer) or \
              grade_answer_sympy(model_answer, gt_answer)  # Line 39

    if not fast:
        # Use stricter latex equality check
        correct = correct or is_latex_equal(model_answer, gt_answer)  # Line 43-46

    return correct  # Line 47
```

**Multi-Method Grading**:
1. String matching with normalization
2. SymPy symbolic equality
3. (Optional) LaTeX parsing and comparison

---

### 4. Reward Function

**File**: `/verl/verl/utils/reward_score/ttrl_math/__init__.py`

```python
def reward_func(data_source, solution_str, ground_truth, ...):  # Line 105
    try:
        res = compute_score(solution_str, str(ground_truth))  # Line 109

        if isinstance(res, dict):
            return res  # Line 112: Return full metrics
        elif isinstance(res, (int, float, bool)):
            return float(res)  # Line 114: Return scalar reward
```

```python
def compute_score(model_response, gt_answer, fast=False):  # Line 67
    model_answer = extract_answer(model_response)  # Line 68

    if model_answer is None:
        return {"score": 0.0, "format_score": 0.0, "acc": False, ...}  # Line 71-77

    # Grade the answer
    is_correct = grade(model_answer, gt_answer, fast)  # Line 83

    if is_correct:
        return {"score": 1.0, "format_score": 1.0, "acc": True, ...}  # Line 89-95
    else:
        return {"score": 0.0, "format_score": 1.0, "acc": False, ...}  # Line 97-103
```

**Reward Structure**:
- **score**: 1.0 if correct, 0.0 if wrong, 0.0 if unparseable
- **format_score**: 1.0 if parseable (has `\boxed{}`), 0.0 otherwise
- **acc**: Boolean correctness

---

## Summary of TTRL Algorithm

### Pseudocode

```
For each training batch:
  1. Load batch of prompts with original ground truth

  2. Generate n_votes_per_prompt (64) samples per prompt
     - Use temperature sampling for diversity

  3. Majority Voting:
     For each prompt:
       - Extract answers from all 64 responses
       - Simplify/normalize answers
       - Count occurrences
       - Select most frequent as pseudo-label
       - Calculate confidence (majority_ratio)

  4. Replace ground_truth with majority-voted labels

  5. Downsample to n_samples_per_prompt (32) samples

  6. Compute rewards using pseudo-labels:
     reward = 1.0 if response matches majority vote else 0.0

  7. Calculate GRPO advantages:
     For each prompt group:
       mean_reward = average of 32 rewards
       std_reward = std of 32 rewards
       advantage = (reward - mean_reward) / std_reward

  8. PPO update:
     Update policy to increase log_prob of high-advantage responses

  9. Metrics (with original labels):
     - Restore original ground truth
     - Compute reward with true labels
     - Log: label_accuracy, majority_ratio, pass@n, etc.
```

---

### Mathematical Formulation

**Majority Vote**:
```
ŷᵢ = argmax_{y} |{r ∈ Rᵢ : extract_answer(r) = y}|

where:
  - Rᵢ = {r₁, r₂, ..., r₆₄} (64 responses for prompt i)
  - ŷᵢ = pseudo-label for prompt i
```

**Reward**:
```
r(rⱼ, ŷᵢ) = {
  1.0  if extract_answer(rⱼ) = ŷᵢ
  0.0  otherwise
}
```

**GRPO Advantage** (for response j of prompt i):
```
Aᵢⱼ = (rᵢⱼ - μᵢ) / σᵢ

where:
  - rᵢⱼ = reward of response j for prompt i
  - μᵢ = mean(rᵢ₁, rᵢ₂, ..., rᵢ₃₂)
  - σᵢ = std(rᵢ₁, rᵢ₂, ..., rᵢ₃₂)
```

**PPO Objective**:
```
L(θ) = E[min(
  ratio(θ) * A,
  clip(ratio(θ), 1-ε, 1+ε) * A
)]

where:
  - ratio(θ) = π_θ(a|s) / π_θ_old(a|s)
  - A = advantage
  - ε = clip range (typically 0.2)
```

---

## File Reference Summary

| Component | File | Key Lines |
|-----------|------|-----------|
| **Data Preprocessing** | `/verl/data/preprocess.py` | 6-33 |
| **TTRL Config** | `/verl/verl/trainer/config/ppo_trainer_ttrl.yaml` | 9-17 |
| **Main Entry** | `/verl/verl/trainer/main_ppo.py` | 29-214 |
| **Training Loop** | `/verl/verl/trainer/ppo/ray_trainer.py` | 1137-1148, 1311-1319 |
| **Majority Voting** | `/verl/verl/trainer/ppo/ttrl_utils.py` | 86-122 |
| **Apply TTRL Labels** | `/verl/verl/trainer/ppo/ttrl_utils.py` | 50-83 |
| **Downsampling** | `/verl/verl/trainer/ppo/ttrl_utils.py` | 20-32 |
| **TTRL Metrics** | `/verl/verl/trainer/ppo/ttrl_utils.py` | 128-214 |
| **Reward Function** | `/verl/verl/utils/reward_score/ttrl_math/__init__.py` | 105-120 |
| **Answer Extraction** | `/verl/verl/utils/reward_score/ttrl_math/__init__.py` | 30-33 |
| **Answer Grading** | `/verl/verl/utils/reward_score/ttrl_math/__init__.py` | 36-47 |
| **GRPO Algorithm** | `/verl/verl/trainer/ppo/core_algos.py` | 205-255 |
| **Training Script** | `/verl/examples/ttrl/Qwen2.5/aime.sh` | All |

---

## Key Insights

1. **TTRL Core Innovation**: Replace ground truth with majority vote (line 78 in `ttrl_utils.py`)

2. **Downsampling Strategy**: Generate more samples for voting than training (64 vs 32) to improve label quality while controlling cost

3. **GRPO Synergy**: GRPO's group-relative advantage naturally fits TTRL's multi-sample-per-prompt setup

4. **Metric Tracking**: Always compute both TTRL reward (with majority labels) and true reward (with original labels) to track progress

5. **Answer Normalization**: Critical for math problems - simplify expressions before comparison to avoid spurious disagreements

6. **Confidence Tracking**: Majority ratio indicates label quality - low ratio suggests uncertain/difficult problems

---

## How to Use This Guide

1. **To understand TTRL**: Read sections 1-3 for overview and methodology
2. **To modify TTRL**: Focus on section 6 (Core TTRL Components)
3. **To debug**: Use section 8 (Detailed Walkthrough) to trace execution
4. **To extend**: Study the file reference table to locate relevant code
5. **To run experiments**: Follow section 9 (Complete Workflow Example)

---

## References

- **Paper**: TTRL: Test-Time Reinforcement Learning (arXiv:2504.16084)
  - https://arxiv.org/abs/2504.16084
- **GitHub**: https://github.com/PRIME-RL/TTRL
- **Base Framework**: VERL (Volcano Engine Reinforcement Learning)
  - https://github.com/volcengine/verl

---

*This guide was created by analyzing the TTRL codebase implementation. All line numbers are accurate as of the current repository state.*
