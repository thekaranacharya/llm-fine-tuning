import torch
from .base_model import BaseModel
from functools import partial

# Define the Adapter layer
class Adapter(torch.nn.Module):
    """
    Implements Adapter layer as described in [https://arxiv.org/pdf/1902.00751]

    Linear -> GELU -> Linear
    """
    def __init__(self, linear_out_dim: int, bottleneck_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(linear_out_dim, bottleneck_dim)  # Feedforward down-project
        self.gelu = torch.nn.GELU()  # non-linearity
        self.linear2 = torch.nn.Linear(bottleneck_dim, linear_out_dim)  # Feedforward up-project

    def forward(self, x):
        residual = x
        x = self.gelu(self.linear1(x))
        x = self.linear2(x)
        return x + residual

# Define the AdaptedLinear layer
class AdaptedLinear(torch.nn.Module):
    """
    Accepts a linear layer and adds an Adapter layer to it
    """
    def __init__(self, linear, bottleneck_dim):
        super().__init__()
        self.linear = linear
        self.adapter = Adapter(linear.out_features, bottleneck_dim)

    def forward(self, x):
        x = self.linear(x)  # Normal linear layer propogation
        return self.adapter(x)  # Adapter layer propogation

# Define the AdaptedModel class
class AdaptedModel(BaseModel):
    """
    Adds Adapter blocks as specified here: [https://arxiv.org/pdf/1902.00751]

    TransformerBlock architecture:
    ```
        TransformerBlock(
            (attention): MultiHeadSelfAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (q_lin): Linear(in_features=768, out_features=768, bias=True)
                (k_lin): Linear(in_features=768, out_features=768, bias=True)
                (v_lin): Linear(in_features=768, out_features=768, bias=True)
                (out_lin): Linear(in_features=768, out_features=768, bias=True)
            )
            (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (ffn): FFN(
                (dropout): Dropout(p=0.1, inplace=False)
                (lin1): Linear(in_features=768, out_features=3072, bias=True)
                (lin2): Linear(in_features=3072, out_features=768, bias=True)
                (activation): GELUActivation()
            )
            (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
    ```

    Adds the Adapter layer in each TransformerBlock at the following positions:
        - after the (out_lin) in (attention)
        - after the (lin2) in (ffn)
    """
    def __init__(
        self, 
        model_uri: str = "distilbert/distilbert-base-uncased", 
        num_classes: int = 2, freeze_all: bool = True,
        bottleneck_dim: int = 4,
    ):
        # Initialise parent class
        super().__init__(
            model_uri=model_uri,
            num_classes=num_classes, 
            freeze_all=freeze_all
        )

        self.bottleneck_dim = bottleneck_dim

        # print(f"[DEBUG]Before adding Adapters:\n{self.get_transformer_block()}\n")
        self.__adapt()
        # print(f"[DEBUG]After adding Adapters:\n{self.get_transformer_block()}\n")

    # def get_transformer_block(self):
    #     """
    #     Method that returns the TransformerBlock architecture
    #     """
    #     return self.model.distilbert.transformer.layer[0]

    def __unfreeze_specific_layers(self) -> None:
        """
        Method that unfreezes specific layers in the model
        """
        print("\n[DEBUG]Unfreezing specific layers...\n")
        # Unfreeze (sa_layer_norm) and (output_layer_norm) in each TransformerBlock
        for block in self.model.distilbert.transformer.layer:
            block.sa_layer_norm.requires_grad = True
            block.output_layer_norm.requires_grad = True
        
        # Unfreeze final classifcation layer
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    
    def __adapt(self) -> None:
        """
        Method that replaces specific Linear layers with AdaptedLinear layers
        """
        print("\n[DEBUG]Adding Adapters...\n")
        adapted_linear = partial(AdaptedLinear, bottleneck_dim=self.bottleneck_dim)

        # Replace all Linear layers within the TransformerBlock with AdaptedLinear
        for block in self.model.distilbert.transformer.layer:
            ## Transformer Block: Multi-head Self-Attention block
            block.attention.out_lin = adapted_linear(block.attention.out_lin)

            ## Transformer Block: Feed-forward block
            block.ffn.lin2 = adapted_linear(block.ffn.lin2)

        # Unfreeze specific layers
        self.__unfreeze_specific_layers()