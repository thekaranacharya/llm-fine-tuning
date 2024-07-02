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

MODEL_NAME = "distilbert/distilbert-base-uncased"
DATASET_NAME = "stanfordnlp/imdb"

# Dataset ETL
dataset_utils = DatasetUtils(
    dataset_uri=DATASET_NAME, model_uri=MODEL_NAME, batch_size=64, num_workers=8
)

train_loader = dataset_utils.get_data_loader("train")
val_loader = dataset_utils.get_data_loader("val")
test_loader = dataset_utils.get_data_loader("test")

test_losses, test_accuracies = {}, {}
train_losses, train_accuracies = {}, {}
training_times, trainable_params = {}, {}

##########################
# 1. Test with BaseModel
base_model = BaseModel(model_uri=MODEL_NAME, num_classes=2, freeze_all=True)

# Test baseline performance on downstream task
test_loss, test_acc = base_model.predict(test_loader)
print(f"Test Loss [Baseline]: {test_loss:.2f}")
print(f"Test accuracy [Baseline]: {test_acc:.2f}%")

# Add to dictionary
test_losses["Baseline"] = round(test_loss, 3)
test_accuracies["Baseline"] = round(test_acc, 3)


##########################
# 2. Test with SimpleFTModel
simple_ft_model = SimpleFTModel()

# Get trainable parameter count
trainable_params_percentage = simple_ft_model.get_parameter_count()
print(f"\n% of trainable parameters: {trainable_params_percentage:.2f} %\n")

start = time.time()
# Train the model
train_loss, train_acc = simple_ft_model.train(
    train_loader, val_loader, num_epochs=1, learning_rate=8e-4
)
training_time = time.time() - start
print(f"Training time: {training_time:.2f} seconds")

# Test performance on downstream task
test_loss, test_acc = simple_ft_model.predict(test_loader)
print(f"Test Loss [Simple Fine-tuning]: {test_loss:.2f}")
print(f"Test accuracy [Simple Fine-tuning]: {test_acc:.2f}%")

# Add to dictionary
test_losses["Simple Fine-tuning"] = round(test_loss, 3)
test_accuracies["Simple Fine-tuning"] = round(test_acc, 3)
train_losses["Simple Fine-tuning"] = round(train_loss, 3)
train_accuracies["Simple Fine-tuning"] = round(train_acc, 3)
training_times["Simple Fine-tuning"] = round(training_time, 3)
trainable_params["Simple Fine-tuning"] = round(trainable_params_percentage, 3)

##########################
# 3. Test with AdaptedModel
adapted_model = AdaptedModel(bottleneck_dim=16)

# Get trainable parameter count
trainable_params_percentage = adapted_model.get_parameter_count()
print(f"\n% of trainable parameters: {trainable_params_percentage:.2f} %\n")

start = time.time()
# Train the model
train_loss, train_acc = adapted_model.train(
    train_loader, val_loader, num_epochs=1, learning_rate=8e-4
)
training_time = time.time() - start
print(f"Training time: {training_time:.2f} seconds")

# Test performance on downstream task
test_loss, test_acc = adapted_model.predict(test_loader)
print(f"Test Loss [Adapted Model]: {test_loss:.2f}")
print(f"Test accuracy [Adapted Model]: {test_acc:.2f}%")

# Add to dictionary
test_losses["Adapter"] = round(test_loss, 3)
test_accuracies["Adapter"] = round(test_acc, 3)
train_losses["Adapter"] = round(train_loss, 3)
train_accuracies["Adapter"] = round(train_acc, 3)
training_times["Adapter"] = round(training_time, 3)
trainable_params["Adapter"] = round(trainable_params_percentage, 3)

##########################
# 4. Test with LoRAModel
lora_model = LoRAModel(lora_rank=16, lora_alpha=16)

# Get trainable parameter count
trainable_params_percentage = lora_model.get_parameter_count()
print(f"\n% of trainable parameters: {trainable_params_percentage:.2f} %\n")

start = time.time()
# Train the model
train_loss, train_acc = lora_model.train(
    train_loader, val_loader, num_epochs=1, learning_rate=2e-5
)
training_time = time.time() - start
print(f"Training time: {training_time:.2f} seconds")

# Test performance on downstream task
test_loss, test_acc = lora_model.predict(test_loader)
print(f"Test Loss [LoRA]: {test_loss:.2f}")
print(f"Test accuracy [LoRA]: {test_acc:.2f}%")

# Add to dictionary
test_losses["LoRA"] = round(test_loss, 3)
test_accuracies["LoRA"] = round(test_acc, 3)
train_losses["LoRA"] = round(train_loss, 3)
train_accuracies["LoRA"] = round(train_acc, 3)
training_times["LoRA"] = round(training_time, 3)
trainable_params["LoRA"] = round(trainable_params_percentage, 3)

# Evaluation
plotting_utils = PlottingUtils()
plotting_utils.plot_training_curves(
    train_losses=train_losses, train_accuracies=train_accuracies
)

plotting_utils.plot_test_curves(
    test_losses=test_losses, test_accuracies=test_accuracies
)

plotting_utils.plot_training_times(training_times=training_times)
plotting_utils.plot_trainable_params(trainable_params=trainable_params)
