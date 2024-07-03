"""
Script that fine-tunes a pre-trained large language model using multiple strategies
on a downstream task (IMDB sentiment classification).
"""

# Imports
from rich import print
import time
from data_utils import DatasetUtils
from models import BaseModel, SimpleFTModel, LoRAModel, AdaptedModel
from plotting_utils import PlottingUtils
import json

MODEL_NAME = "distilbert/distilbert-base-uncased"
DATASET_NAME = "stanfordnlp/imdb"

# Dataset ETL
dataset_utils = DatasetUtils(
    dataset_uri=DATASET_NAME, model_uri=MODEL_NAME, batch_size=64, num_workers=8
)

train_loader = dataset_utils.get_data_loader("train")
val_loader = dataset_utils.get_data_loader("val")
test_loader = dataset_utils.get_data_loader("test")

# Initialise dictionaries to store results
test_losses, test_accuracies = {}, {}
train_losses, train_accuracies = {}, {}
training_times, trainable_params = {}, {}

##########################
# 1. Test with BaseModel
base_model = BaseModel(model_uri=MODEL_NAME, num_classes=2, freeze_all=True)

# Test baseline performance on downstream task
test_loss, test_acc = base_model.predict(test_loader)
print(f"Test Loss [Baseline (No fine-tuning)]: {test_loss:.2f}")
print(f"Test accuracy [Baseline (No fine-tuning)]: {test_acc:.2f}%")

# Add to dictionary
test_losses["Baseline (No fine-tuning)"] = test_loss
test_accuracies["Baseline (No fine-tuning)"] = test_acc

##########################
# 2. Test with SimpleFTModel
simple_ft_model = SimpleFTModel()

# Get trainable parameter count
trainable_params_percentage = simple_ft_model.get_parameter_count()
print(f"\n% of trainable parameters: {trainable_params_percentage:.2f} %\n")

start = time.time()
# Train the model
train_loss, train_acc = simple_ft_model.train(
    train_loader, val_loader, num_epochs=2, learning_rate=8e-4
)
training_time = time.time() - start
print(f"Training time: {training_time:.2f} seconds")

# Test performance on downstream task
test_loss, test_acc = simple_ft_model.predict(test_loader)
print(f"Test loss [Simple]: {test_loss:.2f}")
print(f"Test accuracy [Simple]: {test_acc:.2f}%")

# Add to dictionary
test_losses["Simple"] = test_loss
test_accuracies["Simple"] = test_acc
train_losses["Simple"] = train_loss
train_accuracies["Simple"] = train_acc
training_times["Simple"] = training_time
trainable_params["Simple"] = trainable_params_percentage

##########################
# 3. Test with AdaptedModel
adapted_model = AdaptedModel(bottleneck_dim=16)

# Get trainable parameter count
trainable_params_percentage = adapted_model.get_parameter_count()
print(f"\n% of trainable parameters: {trainable_params_percentage:.2f} %\n")

start = time.time()
# Train the model
train_loss, train_acc = adapted_model.train(
    train_loader, val_loader, num_epochs=2, learning_rate=8e-4
)
training_time = time.time() - start
print(f"Training time: {training_time:.2f} seconds")

# Test performance on downstream task
test_loss, test_acc = adapted_model.predict(test_loader)
print(f"Test loss [Adapter]: {test_loss:.2f}")
print(f"Test accuracy [Adapter]: {test_acc:.2f}%")

# Add to dictionary
test_losses["Adapter"] = test_loss
test_accuracies["Adapter"] = test_acc
train_losses["Adapter"] = train_loss
train_accuracies["Adapter"] = train_acc
training_times["Adapter"] = training_time
trainable_params["Adapter"] = trainable_params_percentage

##########################
# 4. Test with LoRAModel
lora_model = LoRAModel(lora_rank=16, lora_alpha=16)

# Get trainable parameter count
trainable_params_percentage = lora_model.get_parameter_count()
print(f"\n% of trainable parameters: {trainable_params_percentage:.2f} %\n")

start = time.time()
# Train the model
train_loss, train_acc = lora_model.train(
    train_loader, val_loader, num_epochs=2, learning_rate=2e-5
)
training_time = time.time() - start
print(f"Training time: {training_time:.2f} seconds")

# Test performance on downstream task
test_loss, test_acc = lora_model.predict(test_loader)
print(f"Test loss [LoRA]: {test_loss:.2f}")
print(f"Test accuracy [LoRA]: {test_acc:.2f}%")

# Add to dictionary
test_losses["LoRA"] = test_loss
test_accuracies["LoRA"] = test_acc
train_losses["LoRA"] = train_loss
train_accuracies["LoRA"] = train_acc
training_times["LoRA"] = training_time
trainable_params["LoRA"] = trainable_params_percentage

##########################
# Evaluation
## Save results to JSON

with open("results.json", "w") as f:
    json.dump(
        {
            "test_losses": test_losses,
            "test_accuracies": test_accuracies,
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "training_times": training_times,
            "trainable_params": trainable_params,
        },
        f,
    )

## Plotting
plotting_utils = PlottingUtils()
plotting_utils.plot_training_curves(
    train_losses=train_losses, train_accuracies=train_accuracies
)

plotting_utils.plot_test_curves(
    test_losses=test_losses, test_accuracies=test_accuracies
)

plotting_utils.plot_training_times(training_times=training_times)
plotting_utils.plot_trainable_params(trainable_params=trainable_params)
