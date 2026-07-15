#!/usr/bin/env python3
"""
================================================================================
Whisper Small Full Fine-Tuning Script (Production Quality)
Optimized for single 6GB VRAM GPU (e.g., RTX 4050)
================================================================================

HARDWARE ADAPTATIONS & HYPERPARAMETER REASONING:

1. Full Fine-Tuning on 6GB VRAM:
   Full fine-tuning of Whisper Small (~244M parameters) normally requires >12GB VRAM.
   To make this fit on your 6GB RTX 4050 without resorting to PEFT/LoRA, we MUST use:
   - Gradient Checkpointing: Trades compute for memory by dropping intermediate activations.
   - Mixed Precision (FP16): Halves the memory footprint of weights and activations.
   - 8-bit AdamW Optimizer: Standard Adam takes ~2GB for optimizer states. 8-bit AdamW 
     via `bitsandbytes` quantizes these states, reducing footprint to ~500MB with virtually zero loss.

2. Batch Size & Gradient Accumulation:
   - Original: 64
   - Adapted: per_device_train_batch_size = 4, gradient_accumulation_steps = 16
   - Why: A true batch of 64 audio files will instantly cause a CUDA OOM on 6GB. 
     By processing 4 samples at a time and accumulating gradients over 16 steps, 
     we achieve the exact same mathematical effective batch size (4 * 16 = 64) 
     while staying well within your VRAM limit.

3. Learning Rate:
   - Original: 3e-4
   - Adapted: 1e-5
   - Why: 3e-4 is standard for a brand new fine-tuning run from the base English model. 
     Since your model is already fine-tuned on OpenSLR-80, using 3e-4 will likely cause 
     "catastrophic forgetting", destroying the existing weights. 1e-5 is a safe, standard 
     learning rate for continuing fine-tuning.

4. Warmup Steps:
   - Original: 200
   - Adapted: 1000
   - Why: With a massive dataset of 400,000 samples and a batch size of 64, an epoch is 
     ~6,250 steps. 200 warmup steps is practically nothing. 1000 steps (~15% of an epoch) 
     gives the 8-bit optimizer enough time to stabilize variance before fully updating weights.

5. Regularization (Label Smoothing & Weight Decay):
   - Adapted: weight_decay = 0.01, label_smoothing_factor = 0.1
   - Why: For a massive 400k dataset, Whisper can become overconfident in its predictions, 
     leading to higher CER on unseen test data. Label smoothing forces the model to be 
     slightly less confident, dramatically improving robustness against noisy transcripts.

6. Epochs & Early Stopping:
   - Original: 30
   - Adapted: 3 (with Early Stopping patience of 3 evaluations)
   - Why: 30 epochs on 400,000 samples is 12 million samples seen. Whisper will severely 
     overfit long before epoch 30. We set it to 3 epochs with a strict Early Stopping 
     callback monitoring validation WER to capture the absolute lowest error rate.

7. Dataloader Workers:
   - Adapted: dataloader_num_workers = 4
   - Why: You have 24GB of system RAM, which is ample. 4 parallel dataloader workers 
     will keep the RTX 4050 fully saturated without CPU/RAM-bottlenecking.

PREREQUISITES:
    pip install torch transformers datasets evaluate jiwer bitsandbytes tensorboard accelerate
================================================================================
"""

import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# Paths and Models
MODEL_ID = "chuuhtetnaing/whisper-small-myanmar"  # Replace with your actual local/HF checkpoint
DATASET_ID = "your-hf-username/myanmar-audio-dataset" # Replace with your dataset
OUTPUT_DIR = "./whisper-myanmar-finetuned"
LANGUAGE = "my" # ISO code for Burmese
TASK = "transcribe"

# Optimization & Hardware Config
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 16
OPTIMIZER = "adamw_bnb_8bit"
FP16 = True
DATALOADER_NUM_WORKERS = 4

# Training Schedule
LEARNING_RATE = 1e-5
EPOCHS = 3
WARMUP_STEPS = 1000
MAX_STEPS = -1 # -1 means train for all EPOCHS

# Checkpointing & Evaluation Frequency
EVAL_STEPS = 500  # Evaluate every 500 steps
SAVE_STEPS = 500  # Save every 500 steps (keep aligned with eval)
LOGGING_STEPS = 50
SAVE_TOTAL_LIMIT = 3 # Keep only the latest 3 checkpoints to save disk space
EARLY_STOPPING_PATIENCE = 3 # Stop if validation WER doesn't improve for 3 evals

# Regularization
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING_FACTOR = 0.1
GENERATION_MAX_LENGTH = 225

# Audio Features
SAMPLING_RATE = 16000

# ==============================================================================
# 2. DATA PREPARATION
# ==============================================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that dynamically pads the inputs received.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad the audio features to 3000 frames (Whisper standard)
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad the labels to the max length in the batch
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly during training
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step, cut it here since it's appended later
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# ==============================================================================
# 3. MAIN SCRIPT
# ==============================================================================

def main():
    print("=== Starting Whisper Full Fine-Tuning Pipeline ===")
    
    # 3.1: Load Processor, Feature Extractor, and Tokenizer
    print(f"Loading processor for {MODEL_ID}...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)

    # 3.2: Load Dataset
    print(f"Loading dataset {DATASET_ID}...")
    # NOTE: Adjust split names ("train", "validation", "test") according to your dataset structure
    dataset = load_dataset(DATASET_ID)

    # Cast audio column to exactly 16kHz which Whisper expects
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    # Preprocessing function
    def prepare_dataset(batch):
        # Load and resample audio data
        audio = batch["audio"]

        # Compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # Encode target text to label ids 
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    print("Extracting features and tokenizing labels (this may take a while)...")
    # Using multiple processors speeds up the feature extraction step massively
    dataset = dataset.map(
        prepare_dataset, 
        remove_columns=dataset.column_names["train"], 
        num_proc=os.cpu_count() or 4
    )

    # 3.3: Load the Model
    print(f"Loading model {MODEL_ID}...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    
    # Crucial for memory efficiency during full fine-tuning
    model.config.use_cache = False
    
    # Force language and task for generation during evaluation
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    # 3.4: Setup Metrics
    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # Decode tokens
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Basic normalization for cleaner metric calculations
        pred_str = [x.strip() for x in pred_str]
        label_str = [x.strip() for x in label_str]

        # Filter out empty references to avoid division by zero errors in metrics
        filtered_pred_str = []
        filtered_label_str = []
        for p, l in zip(pred_str, label_str):
            if len(l) > 0:
                filtered_pred_str.append(p)
                filtered_label_str.append(l)

        if len(filtered_label_str) == 0:
            return {"wer": 1.0, "cer": 1.0}

        wer = metric_wer.compute(predictions=filtered_pred_str, references=filtered_label_str)
        cer = metric_cer.compute(predictions=filtered_pred_str, references=filtered_label_str)

        return {"wer": wer, "cer": cer}

    # 3.5: Configure Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        num_train_epochs=EPOCHS,
        gradient_checkpointing=True,         # CRITICAL for 6GB VRAM
        fp16=FP16,                           # CRITICAL for 6GB VRAM
        eval_strategy="steps",
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        predict_with_generate=True,
        generation_max_length=GENERATION_MAX_LENGTH,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        report_to=["tensorboard"],           # Enable TensorBoard logging
        load_best_model_at_end=True,         # Required for EarlyStopping
        metric_for_best_model="wer",         # Optimize for lowest WER
        greater_is_better=False,
        push_to_hub=False,
        save_total_limit=SAVE_TOTAL_LIMIT,   # Keep VRAM/Disk usage low
        optim=OPTIMIZER,                     # 8-bit AdamW for VRAM savings
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        weight_decay=WEIGHT_DECAY,
        label_smoothing_factor=LABEL_SMOOTHING_FACTOR,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # 3.6: Initialize Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    # 3.7: Execute Training
    print("\n=== Training Started ===")
    # Automatically resume from latest checkpoint if one exists
    latest_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = os.path.join(OUTPUT_DIR, sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1])
            print(f"Resuming training from checkpoint: {latest_checkpoint}")
    
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    # 3.8: Evaluate Test Set and Save Final Model
    print("\n=== Evaluating on Test Set ===")
    test_results = trainer.predict(dataset["test"])
    print(f"Test WER: {test_results.metrics['test_wer']:.4f}")
    print(f"Test CER: {test_results.metrics['test_cer']:.4f}")

    print("\n=== Saving Final Best Model ===")
    # Save the model, processor, tokenizer, and config to output_dir
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Model saved successfully to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
