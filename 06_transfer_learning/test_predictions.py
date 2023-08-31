import matplotlib.pyplot as plt
import torch
import torchvision
import predict

from torch import nn
from torchvision import transforms
from torchinfo import summary
from pathlib import Path
import torchmetrics, mlxtend

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from tqdm.auto import tqdm
import sys
import os

# Get the parent directory of the current script (06_transfer_learning)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the path to going_modular to sys.path
going_modular_path = os.path.join(parent_dir, 'going_modular')
sys.path.append(going_modular_path)

# Now you can import modules from going_modular
import data_setup

# Setup path to data folder
data_path = Path("../data/")
image_path = data_path / "pizza_steak_sushi_20_percent"

# Setup Dirs
train_dir = image_path / "train"
test_dir = image_path / "test"

# Get a set of pretrained model weights
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet

# Get the transforms used to create our pretrained weights
auto_transforms = weights.transforms()

# Create training and testing DataLoaders as well as get a list of class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=auto_transforms, # perform same data transforms on our own data as the pretrained model
                                                                               batch_size=32) # set mini-batch size to 32
# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

model_0 = predict.load_model("models/06_transfer_learning.pth")
class_names = predict.class_names

# Make predictions on the entire test dataset
test_preds = []
model_0.eval()
with torch.inference_mode():
  # Loop through the batches in the test dataloader
  for X, y in tqdm(test_dataloader):
    X, y = X.to(device), y.to(device)
    # Pass the data through the model
    test_logits = model_0(X)

    # Convert the pred logits to pred probs
    pred_probs = torch.softmax(test_logits, dim=1)

    # Convert the pred probs into pred labels
    pred_labels = torch.argmax(pred_probs, dim=1)

    # Add the pred labels to test preds list
    test_preds.append(pred_labels)

# Concatenate the test preds and put them on the CPU
test_preds = torch.cat(test_preds).cpu()


# Get the truth labels for test dataset
test_truth = torch.cat([y for X, y in test_dataloader])

# Setup confusion matrix instance
confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
confmat_tensor = confmat(preds=test_preds,
                         target=test_truth)

# Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy
    class_names=class_names,
    figsize=(10, 7)
)
plt.show()