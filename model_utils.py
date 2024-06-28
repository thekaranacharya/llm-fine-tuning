"""
Module that holds all utilities modeling-related
"""
from transformers import AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from functools import partial
from typing import Tuple

class BaseModel:
    """
    Base class for a pre-trained frozen model
    """
    def __init__(
        self, 
        model_uri: str = "distilbert/distilbert-base-uncased", 
        num_classes: int = 2,
        freeze_all: bool = True,
    ) -> None:
        self.model_uri = model_uri
        self.num_classes = num_classes
        self.freeze_all = freeze_all

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {self.device}\n")

        # Initialise the model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_uri, num_labels=self.num_classes
        )

        # Freeze all the layers if true
        if self.freeze_all:
            for param in self.model.parameters():
                param.requires_grad = False

    def __get_parameter_count(self) -> Tuple[int]:
        """Method that returns trainable, and total parameter count"""
        total_param_count = sum(p.numel() for p in self.model.parameters())
        trainable_param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return trainable_param_count, total_param_count

    def train(
        self, 
        train_loader, 
        val_loader, 
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
    ):
        """
        Method that trains the model for specified number of epochs with the optimizer
        TODO: Modularise into training and validation loop
        """

        # Get trainable parameter count
        trainable_params, total_params = self.__get_parameter_count()
        print(f"% of trainable parameters: {(trainable_params / total_params * 100):.2f} %")

        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Put model on device
        self.model.to(self.device)

        for epoch in range(num_epochs):
            # Training Phase
            self.model.train()

            train_loss = 0.0
            train_correct = 0
            total_train = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
                inputs = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward pass
                outputs = self.model(inputs, attention_mask=attention_mask, labels=labels)

                # Get loss
                loss = outputs.loss
                
                # Zero the gradients
                optimizer.zero_grad()

                # Backward pass + Step
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                total_train += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # Validation Phase
            self.model.eval()

            val_loss = 0.0
            val_correct = 0
            total_val = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    inputs = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)

                    # Forward pass
                    outputs = self.model(inputs, attention_mask=attention_mask, labels=labels)

                    # Get loss
                    loss = outputs.loss

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.logits, 1)
                    total_val += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Epoch Summary
            train_loss = train_loss / total_train
            val_loss = val_loss / total_val
            train_accuracy = 100 * train_correct / total_train
            val_accuracy = 100 * val_correct / total_val

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    def predict(self, data_loader, which: str = "Testing") -> float:
        """
        Method that performs one forward pass across the data
        Computes and returns accuracy
        No gradient updates

        TODO: Modularise and create a new method def testing_loop()
        """
        # Put model on device
        self.model.to(self.device)

        # Set in evaluation mode
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=which):
                inputs = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward pass
                outputs = self.model(inputs, attention_mask=attention_mask, labels=labels)

                # Evaluate the batch
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Compute accuracy
        accuracy = 100 * correct / total
        return accuracy

class SimpleFTModel(BaseModel):
    """
    Child class of BaseModel
    Keeps the entire network frozen
    Only unfreezes the last 2 layers and trains them
    """
    def __init__(self):
        super().__init__()

        # Unfreeze the parameters in the pre_classifier and classifier layer
        self.__unfreeze_classification_head()

    def __unfreeze_classification_head(self):
        # Unfreeze pre_classifier
        for param in self.model.pre_classifier.parameters():
            param.requires_grad = True

        # Unfreeze classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = True


class LoRA(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank))  # A.shape => (in_dim, rank)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))  # B.shape => (rank, out_dim)
        self.alpha = alpha  # hyperparameter: refers to the degree to which to use "new" knowledge

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)


# Define the LinearLoRA layer
class LinearLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRA(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

class LoRAModel(BaseModel):
    """
    Child class of BaseModel
    Applies Low-rank Adaptation (LoRA) parameter-efficient fine-tuning (PEFT)
    Linear layers are replaced by LinearLoRA layers
    """
    def __init__(
        self, 
        lora_rank: int = 4, 
        lora_alpha: int = 8,
    ):
        super().__init__()
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
    
        # Apply LoRA
        self.__apply_lora()

    def __apply_lora(self):
        """
        Method that applies the LinearLoRA layer to certain Linear layers
        within the network
        """
        print("[DEBUG]Applying LoRA...")
        linear_lora = partial(LinearLoRA, rank=self.lora_rank, alpha=self.lora_alpha)

        # Replace all Linear layers within the TransformerBlock with LinearLoRA
        for block in self.model.distilbert.transformer.layer:
            ## Transformer Block: Multi-head Self-Attention block
            # block.attention.q_lin = linear_lora(block.attention.q_lin)
            # block.attention.k_lin = linear_lora(block.attention.k_lin)
            # block.attention.v_lin = linear_lora(block.attention.v_lin)
            # block.attention.out_lin = linear_lora(block.attention.out_lin)

            # Transformer Block: Feed-forward block
            block.ffn.lin1 = linear_lora(block.ffn.lin1)
            block.ffn.lin2 = linear_lora(block.ffn.lin2)

        # Replace pre_classifier and classifier with LinearLoRA
        self.model.pre_classifier = linear_lora(self.model.pre_classifier)
        self.model.classifier = linear_lora(self.model.classifier)

