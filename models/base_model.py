from transformers import AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from typing import Tuple
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
        self.saved_model_path = f"model_{self.__class__.__name__.lower()}_best.pt"
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

        # Save the model
        torch.save(self.model.state_dict(), self.saved_model_path)

    def get_parameter_count(self) -> Tuple[int]:
        """Method that returns trainable, and total parameter count"""
        total_param_count = sum(p.numel() for p in self.model.parameters())
        trainable_param_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        percentage_trainable = (trainable_param_count / total_param_count) * 100

        return percentage_trainable

    def __training_loop(
        self, train_loader, optimizer, description: str
    ) -> Tuple[float]:
        """Training loop with PyTorch"""
        # Training Phase
        self.model.train()

        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for batch in tqdm(train_loader, desc=description):
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

        train_loss = train_loss / total_train
        train_accuracy = 100 * train_correct / total_train
        return train_loss, train_accuracy

    def __validation_test_loop(
        self, val_test_loader, which: str = "Validation"
    ) -> Tuple[float]:
        """Validation / Testing loop with PyTorch"""
        # Validation Phase
        self.model.eval()

        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for batch in tqdm(val_test_loader, desc=which):
                inputs = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward pass
                outputs = self.model(
                    inputs, attention_mask=attention_mask, labels=labels
                )

                # Get loss
                loss = outputs.loss

                val_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                total_val += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / total_val
        val_accuracy = 100 * val_correct / total_val
        return val_loss, val_accuracy

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 10,
        learning_rate: float = 3e-4,
        es_patience: int = 5,
    ) -> Tuple[float]:
        """
        Method that trains the model for specified number of epochs with the optimizer
        - Add early stopping (es_patience = 5) and dynamic learning rate schedule (patience = 2)
        """
        # Define optimizer
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, patience=2)

        # Put model on device
        self.model.to(self.device)

        # Initialize variables
        best_train_loss, best_train_acc = 0.0, 0.0
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            description = f"Epoch {epoch+1}/{num_epochs} (Train)"
            # Training loop
            train_loss, train_accuracy = self.__training_loop(
                train_loader, optimizer, description
            )

            # Validation loop
            val_loss, val_accuracy = self.__validation_test_loop(val_loader)

            current_lr = learning_rate if epoch == 0 else scheduler.get_last_lr()[0]
            # Epoch Summary
            print(
                f"\nEpoch {epoch+1}/{num_epochs}, LR: {current_lr}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%\n"
            )

            # Update the learning rate scheduler
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                # Update best validation loss
                best_val_loss = val_loss

                # Save the model
                torch.save(self.model.state_dict(), self.saved_model_path)

                # Update best training loss and accuracy as well
                best_train_loss = train_loss
                best_train_acc = train_accuracy

                # Reset epochs without improvement
                epochs_without_improvement = 0
            else:
                # Increment epochs without improvement
                epochs_without_improvement += 1

                # Check for early stopping
                if epochs_without_improvement >= es_patience:
                    print(
                        f"\n[DEBUG]Stopping early! Trained for {epoch + 1} / {num_epochs} epochs.\n"
                    )
                    break

        # Return best training loss and accuracy
        return best_train_loss, best_train_acc

    def predict(self, data_loader, which: str = "Test") -> Tuple[float]:
        """
        Computes and returns loss and accuracy on given (test) data
        """
        # Load the best model
        self.model.load_state_dict(torch.load(self.saved_model_path))

        # Put model on device
        self.model.to(self.device)

        # Run validation/test loop once on the given data loader
        return self.__validation_test_loop(data_loader, which)
