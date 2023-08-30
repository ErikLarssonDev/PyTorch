import torch
import torchvision
import argparse

import model_builder

parser = argparse.ArgumentParser()

parser.add_argument("--image_path",
                    help="target image filepath to predict on")

parser.add_argument("--model_path",
                    default="models/05_going_modular_script_mode_tinyvgg_model.pth",
                    type=str,
                    help="target model to use for prediction filepath")

args = parser.parse_args()

# Setup class names
class_names = ["pizza", "steak", "sushi"]

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get the image path
IMG_PATH = args.image_path
print(f"[INFO] Predicting on {IMG_PATH}")

# Function to load the model
def load_model(filepath=args.model_path):
    # Need to use the same hyperparameters as save model
    model = model_builder.TinyVGG(input_shape=3,
                                  hidden_units=128,
                                  output_shape=3).to(device)
    
    # Load the saved model state dictionary from file
    print(f"[INFO] Loading model from: {filepath}")
    model.load_state_dict(torch.load(filepath))
    return model

# Function to load the model and predict on selected image
def predict_on_image(image_path=IMG_PATH, filepath=args.model_path):
    
    # Load the model
    model = load_model(filepath)

    # Load the image and turn it into torch.float32 (same type as model)
    image = torchvision.io.read_image(str(IMG_PATH)).type(torch.float32)

    # Preprocess the image to get it between 0 and 1
    image = image / 255

    # Resize the image to be on the same size as the model
    trainsform = torchvision.transforms.Resize(size=(64, 64),
                                               antialias=True) # To suppress warning
    image = trainsform(image.unsqueeze(dim=0)) # make sure image has batch dimension (shape: [batch_size, height, width, color_channels])

    # Predict on image
    model.eval()
    with torch.inference_mode():
        # Put image to target device
        image.to(device)

        # Get prediction logits
        pred_logits = model(image)

        # Get prediction probabilities
        pred_probs = torch.softmax(pred_logits, dim=1)

        # Get prediction label
        pred_label = torch.argmax(pred_probs, dim=1)
        pred_label_class = class_names[pred_label]

    print(f"[INFO] Pred class: {pred_label_class}, Pred prob: {pred_probs.max():.3f}")

if __name__ == "__main__":
    predict_on_image()