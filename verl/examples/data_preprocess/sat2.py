# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the SAT2 spatial reasoning dataset to parquet format for TTRL training
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/sat2")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--input_file", default="sat2_test.jsonl", help="Path to the input JSONL file")
    parser.add_argument("--base_image_dir", default=".", help="Base directory for image paths")

    args = parser.parse_args()

    # Expand paths
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Read JSONL file
    data_list = []
    input_path = os.path.expanduser(args.input_file)

    with open(input_path, "r") as f:
        for line in f:
            data_list.append(json.loads(line))

    print(f"Loaded {len(data_list)} examples from {input_path}")

    # Define the preprocessing function
    def make_map_fn(split):
        def process_fn(example, idx):
            # Extract fields from SAT2 format
            conversations = example.pop("conversations")
            image_path = example.pop("image")
            data_source = example.pop("source")

            # Get the question and answer from conversations
            question = None
            answer = None
            for turn in conversations:
                if turn["from"] == "human":
                    question = turn["value"]
                elif turn["from"] == "gpt":
                    answer = turn["value"]

            if question is None or answer is None:
                raise ValueError(f"Missing question or answer in example {idx}")

            # Construct full image path
            full_image_path = os.path.join(args.base_image_dir, image_path)

            # Create structured data format matching VERL's expectations
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,  # Already contains <image> placeholder
                    }
                ],
                "images": [full_image_path],  # List of image paths
                "ability": "spatial_reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer  # Store the correct answer choice text
                },
                "extra_info": {
                    "split": split,
                    "index": f"{data_source}-{idx}",
                    "id": example.get("id", idx),
                },
            }
            return data

        return process_fn

    # Create HuggingFace dataset
    dataset = datasets.Dataset.from_list(data_list)

    # Apply preprocessing
    # For SAT2, we're using the test set, but we'll create both train and test
    # by splitting the data (e.g., 80% train, 20% test for validation)
    # Or you can load from separate files if available

    # For now, let's split the dataset
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    # Apply the mapping function
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    # Save to parquet
    train_output_path = os.path.join(local_dir, "train.parquet")
    test_output_path = os.path.join(local_dir, "test.parquet")

    train_dataset.to_parquet(train_output_path)
    test_dataset.to_parquet(test_output_path)

    print(f"Saved {len(train_dataset)} training examples to {train_output_path}")
    print(f"Saved {len(test_dataset)} test examples to {test_output_path}")

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied data to HDFS: {args.hdfs_dir}")
