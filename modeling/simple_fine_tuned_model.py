"""
Implements traditional simple parameter-efficient fine-tuning (PEFT)
Keeps the entire network frozen except the last 2 layers
"""

from .base_model import BaseModel


class SimpleFTModel(BaseModel):
    def __init__(
        self,
        model_uri: str = "distilbert/distilbert-base-uncased",
        num_classes: int = 2,
        freeze_all: bool = True,
    ) -> None:
        """
        Args:
            model_uri: str
                URI of the pre-trained model
            num_classes: int
                Number of classes in the dataset
            freeze_all: bool
                Flag to freeze all the layers of the model
        """
        super().__init__(
            model_uri=model_uri, num_classes=num_classes, freeze_all=freeze_all
        )

        # Unfreeze the parameters in the pre_classifier and classifier layer
        self.__unfreeze_classification_head()

    def __unfreeze_classification_head(self) -> None:
        """
        Method that unfreezes the classification head (last 2 layers of the model)
        """
        print("\n[DEBUG]Unfreezing classification head...\n")
        # Unfreeze pre_classifier
        for param in self.model.pre_classifier.parameters():
            param.requires_grad = True

        # Unfreeze classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = True
