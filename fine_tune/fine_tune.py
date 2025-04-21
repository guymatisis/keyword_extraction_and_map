# fine_tune_keybart.py
#
# Minimal end‑to‑end script to fine‑tune Bloomberg’s “KeyBART” model on a
# custom keyphrase‑generation dataset (text → comma‑separated keyphrases).
# Requires:  transformers >= 4.40, datasets, accelerate, sentencepiece, torch.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import comet_ml
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from scripts.evaluate_model import make_compute_metrics
from config import Config
from trainer import KeyphraseTrainer

os.environ["COMET_API_KEY"] = "AUxlSrpcjGvzxsJLtGl2oMBi5"

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print the configuration if needed
    print(OmegaConf.to_yaml(cfg))
    
    # Convert Hydra config to our configuration class
    config = Config.from_hydra_config(cfg)
    
    # Initialize trainer
    trainer = KeyphraseTrainer(config)
    
    # Train and evaluate
    metrics = trainer.train(compute_metrics_fn=make_compute_metrics(trainer.tokenizer))
    return metrics

if __name__ == "__main__":
    main()
