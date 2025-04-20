# fine_tune_keybart.py
#
# Minimal end‑to‑end script to fine‑tune Bloomberg’s “KeyBART” model on a
# custom keyphrase‑generation dataset (text → comma‑separated keyphrases).
# Requires:  transformers >= 4.40, datasets, accelerate, sentencepiece, torch.

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
import transformers
print('Transformers module path:', transformers.__file__)
import argparse
import torch
import datetime
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune KeyBART on keyphrase data.")
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--single_batch', action='store_true', help='Run a single batch for a quick py')
    return parser.parse_args()


def preprocess_batch(tokenizer, max_in, max_out):
    def preprocess(batch):
        # Alert if any input is longer than max_in
        for i, text in enumerate(batch["text"]):
            num_tokens = len(tokenizer.tokenize(text))
            if isinstance(text, str) and num_tokens > max_in:
                print(f"[Warning] Input at index {i} is {num_tokens} tokens. max_in is ({max_in} tokens): truncated.")
        model_inputs = tokenizer(
            batch["text"],
            max_length=max_in,
            truncation=True,
            padding="max_length",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["keyphrases"],
                max_length=max_out,
                truncation=True,
                padding="max_length",
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return preprocess


def main():
    args_cli = parse_args()

    # ----------------------------------------------------------------------
    # 1. Load checkpoint & tokenizer from the Hugging Face Hub
    # ----------------------------------------------------------------------
    MODEL_NAME = "bloomberg/KeyBART"          # or your own upstream checkpoint
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(args_cli.device)

    # ----------------------------------------------------------------------
    # 2. Read your data  ----------------------------------------------------
    # CSV layout:  text  ,  keyphrases
    # keyphrases column should be a single string:  "deep learning; model compression; pruning"
    # ----------------------------------------------------------------------
    data_dir = os.path.join("data", 'hands-on_machine_learning_with_scikit-learn_keras_and_tensorflow', "processed_inputs")
    dataset = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(data_dir, "train.csv"),
            "validation": os.path.join(data_dir, "valid.csv"),
            "test": os.path.join(data_dir, "test.csv"),
        },
        sep = "|",   # CSV separator
        column_names=["text", "keyphrases"],
    )

    # ----------------------------------------------------------------------
    # 3. Tokenisation / formatting  ----------------------------------------
    #    KeyBART expects <text> → <kp1 ; kp2 ; kp3>
    # ----------------------------------------------------------------------
    max_in   = 1024          # truncate to model context window
    max_out  = 64            # enough for ~10 keyphrases

    preprocess = preprocess_batch(tokenizer, max_in, max_out)

    tokenised = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,   # keep only tensors
        desc="Tokenising"
    )

    # Single batch mode: reduce dataset size and epochs for a quick check
    if args_cli.single_batch:
        for split in ["train", "validation", "test"]:
            tokenised[split] = tokenised[split].select(range(1))
        num_epochs = 1
        batch_size = 1
        grad_accum = 1
    else:
        num_epochs = args_cli.epochs
        batch_size = 4
        grad_accum = 8

    # ----------------------------------------------------------------------
    # 4. Training setup  ----------------------------------------------------
    #    Adjust batch_size & learning_rate to your hardware.
    #    For consumer‑level GPUs use fp16 & gradient accumulation.
    # ----------------------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir="keybart_finetuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,   # effective batch 32
        learning_rate=5e-5,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=max_out,
        fp16=(args_cli.device == 'cuda'),
        report_to="none",  # disables wandb
        load_best_model_at_end=True,  # load the best model when finished training
        metric_for_best_model="eval_loss",  # use eval loss to identify the best model
        greater_is_better=False,  # we want to minimize the loss
              )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Stop if no improvement for 2 epochs
    )

    # ----------------------------------------------------------------------
    # 5. Launch fine‑tuning  ------------------------------------------------
    trainer.train()

    # ----------------------------------------------------------------------
    # 6. Optional: evaluate and push to Hub  -------------------------------
    metrics = trainer.evaluate(tokenised["test"], max_length=max_out)
    if metrics is None: print("No metrics returned.")
    print(metrics)

    # Save model with unique directory name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    epoch_str = "0" if args_cli.single_batch else str(num_epochs)
    save_dir = os.path.join("model", f"keybart_finetuned_e{epoch_str}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")
    # trainer.push_to_hub("your‑username/keybart‑finetuned")

if __name__ == "__main__":
    main()
