# model_utils.py
import os
from typing import Dict, Tuple

import numpy as np
import torch
from datasets import DatasetDict
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from config import cfg
from utils import ensure_dir


def get_model_name(backbone: str) -> Tuple[str, str]:
    if backbone == "biobert":
        return cfg.BIOBERT_MODEL, cfg.OUTPUT_DIR_BIOBERT
    elif backbone == "pubmedbert":
        return cfg.PUBMEDBERT_MODEL, cfg.OUTPUT_DIR_PUBMEDBERT
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )


def compute_macro_metrics(preds, labels, id2label: Dict[int, str]) -> Dict[str, float]:
    y_true = labels
    y_pred = preds
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def train_supervised_model(
    ds_rel: DatasetDict,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    backbone: str,
    device: torch.device,
    max_train_examples: int = 20000,
    max_eval_examples: int = 5000,
) -> None:
    print(f"DEBUG: Training backbone={backbone} on device={device}", flush=True)
    model_name, output_dir = get_model_name(backbone)
    print(f"DEBUG: model_name={model_name}", flush=True)
    print(f"DEBUG: output_dir={output_dir}", flush=True)

    ensure_dir(output_dir)

    num_labels = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Subset for speed if requested
    train_dataset = ds_rel["train"]
    eval_dataset = ds_rel["validation"]

    if max_train_examples is not None and max_train_examples < len(train_dataset):
        print(
            f"DEBUG: using subset of train: {max_train_examples} examples",
            flush=True,
        )
        train_dataset = train_dataset.select(range(max_train_examples))
    if max_eval_examples is not None and max_eval_examples < len(eval_dataset):
        print(
            f"DEBUG: using subset of validation: {max_eval_examples} examples",
            flush=True,
        )
        eval_dataset = eval_dataset.select(range(max_eval_examples))

    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )

    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    eval_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=1,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        metrics = compute_macro_metrics(preds, labels, id2label)
        return {
            "loss": float(np.mean(np.max(logits, axis=-1))),  
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # ---- TRAIN ----
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Training completed for {backbone}. Model saved at: {output_dir}")
    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)


def evaluate_supervised_model(
    ds_rel: DatasetDict,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    backbone: str,
    device: torch.device,
) -> None:
    print(f"DEBUG: Evaluating backbone={backbone} on device={device}", flush=True)
    model_name, output_dir = get_model_name(backbone)
    print(f"DEBUG: loading model from: {output_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        output_dir,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    test_dataset = ds_rel["test"]
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "eval_tmp"),
        per_device_eval_batch_size=16,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_macro_metrics(preds, labels, id2label)

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    print(f"Test macro metrics for {backbone}:", metrics)

    # Detailed classification report
    preds_logits, labels, _ = trainer.predict(test_dataset)
    preds = np.argmax(preds_logits, axis=-1)
    label_names = [id2label[i] for i in sorted(id2label.keys())]
    print(
        classification_report(
            labels,
            preds,
            target_names=label_names,
            zero_division=0,
        )
    )
