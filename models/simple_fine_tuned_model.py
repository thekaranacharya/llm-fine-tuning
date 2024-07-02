from .base_model import BaseModel

class SimpleFTModel(BaseModel):
    """
    Child class of BaseModel
    Keeps the entire network frozen
    Only unfreezes the last 2 layers and trains them
    """
    def __init__(self) -> None:
        super().__init__()

        # Unfreeze the parameters in the pre_classifier and classifier layer
        self.__unfreeze_classification_head()

    def __unfreeze_classification_head(self) -> None:
        print("\n[DEBUG]Unfreezing classification head...\n")
        # Unfreeze pre_classifier
        for param in self.model.pre_classifier.parameters():
            param.requires_grad = True

        # Unfreeze classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = True