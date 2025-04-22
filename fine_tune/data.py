from dataclasses import dataclass
from typing import Optional, List, Dict
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import os

@dataclass
class BaseDataset:
    data_dir: str
    train_file: str
    validation_file: str
    test_file: Optional[str] = None
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
        data_path = os.path.join(project_root, self.data_dir, self.dataset_name, "page_text", "processed_inputs")
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
class FastAIDataset(BaseDataset):
    dataset_name: str = "deep-learning-for-coders-with-fastai-and-pytorch-first-edition"

    def load(self) -> DatasetDict:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, self.data_dir, self.dataset_name, "page_text", "processed_inputs")
        return load_dataset(
            "csv",
            data_files={
                "train": os.path.join(data_path, self.train_file),
                "validation": os.path.join(data_path, self.validation_file),
            },
            sep=self.data_separator,
        )

@dataclass
class TransformersDataset(BaseDataset):
    dataset_name: str = "natural-language-processing-with-transformers-revised-edition"

    def load(self) -> DatasetDict:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, self.data_dir, self.dataset_name, "page_text", "processed_inputs")
        return load_dataset(
            "csv",
            data_files={
                "train": os.path.join(data_path, self.train_file),
                "validation": os.path.join(data_path, self.validation_file),
            },
            sep=self.data_separator,
        )

@dataclass
class MLPatternsDataset(BaseDataset):
    dataset_name: str = "pub_machine-learning-design-patterns-solutions-to-common-challenges-in-data-preparation-model-building-and-mlops"

    def load(self) -> DatasetDict:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, self.data_dir, self.dataset_name, "page_text", "processed_inputs")
        return load_dataset(
            "csv",
            data_files={
                "train": os.path.join(data_path, self.train_file),
                "validation": os.path.join(data_path, self.validation_file),
            },
            sep=self.data_separator,
        )

@dataclass
class CombinedDataset:
    datasets: List[Dict]
    
    def load(self) -> DatasetDict:
        dataset_instances = []
        dataset_map = {
            'data.HandsOnMLDataset': HandsOnMLDataset,
            'data.FastAIDataset': FastAIDataset,
            'data.TransformersDataset': TransformersDataset,
            'data.MLPatternsDataset': MLPatternsDataset
        }
        
        print(f"Number of datasets in config: {len(self.datasets)}")
        for dataset_config in self.datasets:
            print(f"\nTrying to load dataset with target: {dataset_config.get('_target_')}")
            dataset_class = dataset_map.get(dataset_config.get('_target_'))
            print(f"Found class: {dataset_class}")
            if dataset_class:
                dataset = dataset_class(
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
            # Only merge splits that exist in both datasets
            available_splits = set(additional.keys()) & set(combined.keys())
            for split in available_splits:
                combined[split] = concatenate_datasets([combined[split], additional[split]])
        
        return combined
