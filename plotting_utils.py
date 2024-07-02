import matplotlib.pyplot as plt
import os


class PlottingUtils:
    def __init__(self, dir_path: str = "plots"):
        # Make directory to save plots if it doesn't exist
        self.dir_path = dir_path

        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

    def plot_training_curves(self, train_losses: dict, train_accuracies: dict):
        """
        Method that makes 2 bar plots:
        one for loss and one for accuracy
        Saves both the plots
        """
        # Plot the training loss
        plt.figure(figsize=(10, 6))
        plt.bar(train_losses.keys(), train_losses.values(), color="skyblue")
        plt.xlabel("Strategies")
        plt.ylabel("Training Loss")
        plt.title("Training Loss for each strategy")
        plt.savefig(f"{self.dir_path}/training_loss.png")
        print("[DEBUG]Training loss plot saved.")

        # Plot the training accuracy
        plt.figure(figsize=(10, 6))
        plt.bar(train_accuracies.keys(), train_accuracies.values(), color="orange")
        plt.xlabel("Strategies")
        plt.ylabel("Training Accuracy")
        plt.title("Training Accuracy for each strategy")
        plt.savefig(f"{self.dir_path}/training_accuracy.png")
        print("[DEBUG]Training accuracy plot saved.")

    def plot_test_curves(self, test_losses: dict, test_accuracies: dict):
        """
        Method to plot the test curves
        Makes 2 bar plots: one for loss and one for accuracy
        Saves both the plots
        """
        # Plot the test loss
        plt.figure(figsize=(10, 6))
        plt.bar(test_losses.keys(), test_losses.values(), color="skyblue")
        plt.xlabel("Strategies")
        plt.ylabel("Test Loss")
        plt.title("Test Loss for each strategy")
        plt.savefig(f"{self.dir_path}/test_loss.png")
        print("[DEBUG]Test loss plot saved.")

        # Plot the test accuracy
        plt.figure(figsize=(10, 6))
        plt.bar(test_accuracies.keys(), test_accuracies.values(), color="orange")
        plt.xlabel("Strategies")
        plt.ylabel("Test Accuracy")
        plt.title("Test Accuracy for each strategy")
        plt.savefig(f"{self.dir_path}/test_accuracy.png")
        print("[DEBUG]Test accuracy plot saved.")

    def plot_training_times(self, training_times: dict):
        """
        Method to plot the training times
        Makes a bar plot and saves it
        """
        plt.figure(figsize=(10, 6))
        plt.bar(training_times.keys(), training_times.values(), color="green")
        plt.xlabel("Strategies")
        plt.ylabel("Training Time (s)")
        plt.title("Training Time for each strategy")
        plt.savefig(f"{self.dir_path}/training_times.png")
        print("[DEBUG]Training times plot saved.")

    def plot_trainable_params(self, trainable_params: dict):
        """
        Method to plot the % of trainable parameters
        Makes a bar plot and saves it
        """
        plt.figure(figsize=(10, 6))
        plt.bar(trainable_params.keys(), trainable_params.values(), color="red")
        plt.xlabel("Strategies")
        plt.ylabel("% of Trainable Parameters")
        plt.title("% of Trainable Parameters for each strategy")
        plt.savefig(f"{self.dir_path}/trainable_params.png")
        print("[DEBUG]Trainable params plot saved.")
