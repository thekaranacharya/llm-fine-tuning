import torch
from .base_model import BaseModel
from functools import partial


# Define the LoRA layer
class LoRA(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(
            torch.randn(in_dim, rank)
        )  # A.shape => (in_dim, rank)
        self.B = torch.nn.Parameter(
            torch.zeros(rank, out_dim)
        )  # B.shape => (rank, out_dim)
        self.alpha = alpha  # hyperparameter: refers to the degree to which to use "new" knowledge
        self.rank = rank

    def forward(self, x):
        return (self.alpha / self.rank) * (x @ self.A @ self.B)


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
        model_uri: str = "distilbert/distilbert-base-uncased",
        num_classes: int = 2,
        freeze_all: bool = True,
        lora_rank: int = 4,
        lora_alpha: int = 8,
    ) -> None:
        # Initialise parent class
        super().__init__(
            model_uri=model_uri, num_classes=num_classes, freeze_all=freeze_all
        )
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Apply LoRA
        self.__apply_lora()

    def __apply_lora(self) -> None:
        """
        Method that applies the LinearLoRA layer to certain Linear layers
        within the network

        Only adapt the linear layers in the Attention block as done in
        [https://arxiv.org/pdf/2106.09685], keep MLP frozen
        """
        print("\n[DEBUG]Applying LoRA...\n")
        linear_lora = partial(LinearLoRA, rank=self.lora_rank, alpha=self.lora_alpha)

        # Replace all Linear layers within the TransformerBlock with LinearLoRA
        for block in self.model.distilbert.transformer.layer:
            ## Transformer Block: Multi-head Self-Attention block
            block.attention.q_lin = linear_lora(block.attention.q_lin)
            block.attention.k_lin = linear_lora(block.attention.k_lin)
            block.attention.v_lin = linear_lora(block.attention.v_lin)
            block.attention.out_lin = linear_lora(block.attention.out_lin)
