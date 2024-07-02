# Imports
from rich import print
from data_utils import DatasetUtils
from models import BaseModel, SimpleFTModel, LoRAModel, AdaptedModel

MODEL_NAME = "distilbert/distilbert-base-uncased"
DATASET_NAME = "stanfordnlp/imdb"

# ETL the dataset
dataset_utils = DatasetUtils(
    dataset_uri=DATASET_NAME,
    model_uri=MODEL_NAME,
    batch_size=64,
    num_workers=8
)

train_loader = dataset_utils.get_data_loader("train")
val_loader = dataset_utils.get_data_loader("val")
test_loader = dataset_utils.get_data_loader("test")

##########################
# 1. Test with BaseModel
base_model = BaseModel(
    model_uri=MODEL_NAME,
    num_classes=2,
    freeze_all=True
)

# Test baseline performance on downstream task
test_loss, test_accuracy = base_model.predict(test_loader)
print(f"Test Loss [Baseline]: {test_loss:.2f}")
print(f"Test accuracy [Baseline]: {test_accuracy:.2f}%")


##########################
# 2. Test with SimpleFTModel
simple_ft_model = SimpleFTModel()

# Train the model
simple_ft_model.train(
    train_loader,
    val_loader,
    num_epochs=10,
)

# Test performance on downstream task
test_loss, test_accuracy = simple_ft_model.predict(test_loader)
print(f"Test Loss [Simple Fine-tuning]: {test_loss:.2f}")
print(f"Test accuracy [Simple Fine-tuning]: {test_accuracy:.2f}%")

##########################
# 3. Test with AdaptedModel
adapted_model = AdaptedModel(
    bottleneck_dim=8
)

# Train the model
adapted_model.train(
    train_loader,
    val_loader,
    num_epochs=10,
)

# Test performance on downstream task
test_loss, test_accuracy = adapted_model.predict(test_loader)
print(f"Test Loss [Adapted Model]: {test_loss:.2f}")
print(f"Test accuracy [Adapted Model]: {test_accuracy:.2f}%")

##########################
# 4. Test with LoRAModel
lora_model = LoRAModel(
    lora_rank=8,
    lora_alpha=4
)

# Train the model
lora_model.train(
    train_loader,
    val_loader,
    num_epochs=10,
)

# Test performance on downstream task
test_loss, test_accuracy = lora_model.predict(test_loader)
print(f"Test Loss [LoRA]: {test_loss:.2f}")
print(f"Test accuracy [LoRA]: {test_accuracy:.2f}%")