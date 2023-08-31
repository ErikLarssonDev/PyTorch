from pathlib import Path
from typing import List, Tuple
import torch
import torch
import torchvision

from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

# Get the parent directory of the current script (06_transfer_learning)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the path to going_modular to sys.path
going_modular_path = os.path.join(parent_dir, 'going_modular')
sys.path.append(going_modular_path)

# Now you can import modules from going_modular
import predict

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup class names
class_names = ["pizza", "steak", "sushi"]

# Setup path to data folder
data_path = Path("../data/")
image_path = data_path / "pizza_steak_sushi_20_percent"

# Setup Dirs
train_dir = image_path / "train"
test_dir = image_path / "test"

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str, 
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device=device):
    
    
    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ### 

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability 
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)
    plt.show()

# Function to load the model
def load_model(filepath):
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in model.features.parameters():
        param.requires_grad = False

    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1280, 
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(device)
    
    # Load the saved model state dictionary from file
    print(f"[INFO] Loading model from: {filepath}")
    model.load_state_dict(torch.load(filepath))
    return model

# Get a random list of image paths from test set
import random
num_images_to_plot = 3
test_image_path_list = list(Path(test_dir).glob("*/*.jpg")) # get list all image paths from test data 
test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                       k=num_images_to_plot) # randomly select 'k' image paths to pred and plot
if __name__ == "__main__":
    model = load_model("models/06_transfer_learning.pth")
   
    # Make predictions on and plot the images
    for image_path in test_image_path_sample:
        pred_and_plot_image(model=model, 
                            image_path=image_path,
                            class_names=class_names,
                            # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
                            image_size=(224, 224))