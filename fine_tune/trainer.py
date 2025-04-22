from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    LEDTokenizer,
     LEDForConditionalGeneration
)
from datasets import load_dataset, Dataset, DatasetDict
import os
import datetime
from typing import Dict, Any, Callable
import torch
from config import Config
from data import CombinedDataset

class KeyphraseTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.set_tokenizer()
        self.model = None
        self.trainer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self):
        """Load the model and tokenizer"""
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model.name)
        self.model.to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))

        return self.model, self.tokenizer
    
    def set_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        self.tokenizer.add_tokens(["<largefont>", "</largefont>"])


    def preprocess_batch(self):
        """Create preprocessing function"""
        def preprocess(batch):
            # Alert if any input is longer than max_in
            for i, text in enumerate(batch["text"]):
                num_tokens = len(self.tokenizer.tokenize(text))
                if isinstance(text, str) and num_tokens > self.config.model.max_input_length:
                    print(f"[Warning] Input at index {i} is {num_tokens} tokens. max_in is ({self.config.model.max_input_length} tokens): truncated.")
            
            model_inputs = self.tokenizer(
                batch["text"],
                max_length=self.config.model.max_input_length,
                truncation=True,
                padding="max_length",
            )
            
            if self.config.model.name == "allenai/led-base-16384":
                model_inputs["global_attention_mask"] = [[1] + [0] * (len(input_ids) - 1) for input_ids in model_inputs["input_ids"]]
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        batch["keyphrases"],
                        max_length=self.config.model.max_output_length,
                        truncation=True,
                        padding="max_length",
                    )
                model_inputs["labels"] = labels["input_ids"]

            return model_inputs
        return preprocess
    
    def load_and_process_data(self):
        """Load and preprocess the dataset"""
        # Create and load combined datasets
        combined_dataset = CombinedDataset(datasets=self.config.datasets.datasets)
        dataset = combined_dataset.load()
        
        tokenized = dataset.map(
            self.preprocess_batch(),
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing",
        )
        
        if self.config.debug.single_batch:
            for split in ["train", "validation", "test"]:
                tokenized[split] = tokenized[split].select(range(1))
                
        return tokenized
    
    def setup_trainer(self, tokenized_datasets, compute_metrics_fn: Callable):
        """Set up the trainer with all arguments"""
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output.dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_stategy='epoch',
            save_total_limit=self.config.output.save_total_limit,
            logging_strategy="epoch",
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            predict_with_generate=True,
            generation_max_length=self.config.model.max_output_length,
            fp16=(self.device == 'cuda'),  # Enable fp16 only when using GPU
            report_to="comet_ml",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            run_name=self.config.run_name,
        )
        
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, self.model),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.training.early_stopping_patience)],
            compute_metrics=compute_metrics_fn,
        )
        
        return self.trainer
    
    def train(self, compute_metrics_fn: Callable):
        """Execute the complete training pipeline"""
        # Load model and tokenizer
        self.load_model()
        
        # Load and process data
        tokenized_datasets = self.load_and_process_data()
        
        # Setup trainer
        self.setup_trainer(tokenized_datasets, compute_metrics_fn)
        
        # Train
        self.trainer.train()
        
        # Evaluate
        metrics = self.trainer.evaluate(tokenized_datasets["test"], 
                                     max_length=self.config.model.max_output_length)
        if metrics is None:
            print("No metrics returned.")
        print(metrics)
        
        # Save model
        self.save_model()
        self.trainer.save_model()  # <- Hugging Face version (equivalent to self.model.save_pretrained(...))

        
        return metrics
    
    def save_model(self):
        """Save the model with a unique name"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        epoch_str = "0" if self.config.debug.single_batch else str(self.config.training.num_epochs)
        save_dir = os.path.join("model", f"keybart_finetuned_e{epoch_str}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")
