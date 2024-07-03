"""
Dataset ETL Utilities
References:
- https://lightning.ai/lightning-ai/studios/code-lora-from-scratch?tab=overview
"""

# Imports
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import transformers


class CustomDataset(Dataset):
    def __init__(self, dataset_dict: DatasetDict, partition: str = "train"):
        self.data = dataset_dict[partition]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DatasetUtils:
    def __init__(
        self,
        dataset_uri: str = "stanfordnlp/imdb",
        model_uri: str = "distilbert/distilbert-base-uncased",
        batch_size: int = 64,
        num_workers: int = 8,
        seed: int = 42,
    ) -> None:
        """
        Args:
            dataset_uri: str
                Huggingface URI of the dataset to be used
            model_uri: str
                URI of the pre-trained model to be used
            batch_size: int
                Batch size for the dataloaders
            num_workers: int
                Number of workers for the dataloaders
            seed: int
                Seed for reproducibility
        """
        self.dataset_uri = dataset_uri
        self.model_uri = model_uri
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.dataset = None
        self.tokenized_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Setup
        self.__setup()

    def __load(self):
        """
        Method to load the dataset from Huggingface
        and split into train, val, test
        """
        # Load
        print("[DEBUG]Loading the dataset...")
        dataset = load_dataset(self.dataset_uri, split="train")

        # Split data into 80-10-10 (train-val-test)
        # Shuffled by default
        print("[DEBUG]Splitting the dataset...")
        ## 1. Split into train-val (90%) and test (10%)
        trainval_test = dataset.train_test_split(
            test_size=0.1
        )  # trainval_test["train"] -> Train and val; trainval_test["test"] -> Test

        ## 2. Split train-val into train (90%) and val (10%)
        train_val = trainval_test["train"].train_test_split(
            test_size=0.1
        )  # train_val["train"] -> Train; train_val["test"] -> val

        self.dataset = DatasetDict(
            {
                "train": train_val["train"],
                "val": train_val["test"],
                "test": trainval_test["test"],
            }
        )

    def __tokenize(self):
        """Method to tokenize the dataset in batches"""

        def __batch_tokenize(batch):
            """Method that tokenizes a batch of data"""
            return self.tokenizer(batch["text"], truncation=True, padding="max_length")

        print("[DEBUG]Tokenizing the dataset...")
        # Setup the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_uri)

        self.tokenized_dataset = self.dataset.map(__batch_tokenize, batched=True)
        self.tokenized_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )

    def __setup_dataloaders(self):
        """Method to setup all the dataloaders"""

        print("[DEBUG]Setting up the dataloaders...")
        # Define dataset objects for each partition
        train_tokenized_dataset = CustomDataset(
            self.tokenized_dataset, partition="train"
        )
        val_tokenized_dataset = CustomDataset(self.tokenized_dataset, partition="val")
        test_tokenized_dataset = CustomDataset(self.tokenized_dataset, partition="test")

        # Define the dataloaders for each partition
        self.train_loader = DataLoader(
            train_tokenized_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        self.val_loader = DataLoader(
            val_tokenized_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        self.test_loader = DataLoader(
            test_tokenized_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def __setup(self):
        """
        Method to setup the dataset and dataloaders
        """
        # Set seeds
        transformers.set_seed(self.seed)

        # Load
        self.__load()

        # Tokenize
        self.__tokenize()

        # Setup data loaders
        self.__setup_dataloaders()
        print("[DEBUG]Data setup complete.")

    def get_data_loader(self, which: str = "train"):
        """
        Method to get the dataloader for the specified partition

        Args:
            which: str
                Partition to get the dataloader for
                One of ("train", "val", "test")

        Returns:
            DataLoader: DataLoader for the specified partition
        """
        allowed = ("train", "val", "test")
        if which not in allowed:
            raise ValueError(
                f"Invalid value '{which}' received. Supported one of ({allowed})."
            )

        match which:
            case "train":
                return self.train_loader

            case "val":
                return self.val_loader

            case "test":
                return self.test_loader
