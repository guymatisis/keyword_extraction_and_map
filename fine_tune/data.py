from dataclasses import dataclass
from typing import Optional, List, Dict
from datasets import load_dataset, Dataset, DatasetDict
import os

@dataclass
class BaseDataset:
    data_dir: str
    train_file: str
    validation_file: str
    test_file: str
    text_column: str = "text"
    keyphrases_column: str = "keyphrases"
    data_separator: str = "|"

    def load(self) -> DatasetDict:
        raise NotImplementedError

@dataclass
class HandsOnMLDataset(BaseDataset):
    dataset_name: str = "hands-on_machine_learning_with_scikit-learn_keras_and_tensorflow"

    def load(self) -> DatasetDict:
        # Get the project root directory (parent of fine_tune folder)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, self.data_dir, self.dataset_name, "processed_inputs")
        return load_dataset(
            "csv",
            data_files={
                "train": os.path.join(data_path, self.train_file),
                "validation": os.path.join(data_path, self.validation_file),
                "test": os.path.join(data_path, self.test_file),
            },
            sep=self.data_separator,
        )

@dataclass
class CombinedDataset:
    datasets: List[Dict]
    
    def load(self) -> DatasetDict:
        dataset_instances = []
        for dataset_config in self.datasets:
            if dataset_config.get('_target_') == 'data.HandsOnMLDataset':
                dataset = HandsOnMLDataset(
                    data_dir=dataset_config['data_dir'],
                    dataset_name=dataset_config['dataset_name'],
                    train_file=dataset_config['train_file'],
                    validation_file=dataset_config['validation_file'],
                    test_file=dataset_config['test_file'],
                    text_column=dataset_config['text_column'],
                    keyphrases_column=dataset_config['keyphrases_column'],
                    data_separator=dataset_config['data_separator']
                )
                dataset_instances.append(dataset)
        
        if not dataset_instances:
            raise ValueError("No valid datasets found in configuration")
            
        # Load first dataset as base
        combined = dataset_instances[0].load()
        
        # Merge additional datasets
        for dataset in dataset_instances[1:]:
            additional = dataset.load()
            for split in ["train", "validation", "test"]:
                if split in additional:
                    combined[split] = Dataset.concatenate([combined[split], additional[split]])
        
        return combined
