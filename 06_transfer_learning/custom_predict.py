
from pathlib import Path
import predict
# Download custom image
import requests

# Setup path to data folder
data_path = Path("../data/")

# Setup custom image path
custom_image_path = data_path / "custom_image_steak.jpg"

# Download the image if it doesn't already exist
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        # When downloading from GitHub, need to use the "raw" file link
        # request = requests.get("https://fiskbilen.se/wp-content/uploads/2020/09/hemgjord-sushi-1080x810.jpg")
        request = requests.get("https://whitneybond.com/wp-content/uploads/2021/06/steak-marinade-13.jpg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")

model = predict.load_model("models/06_transfer_learning.pth")
# Pred on our custom image
predict.pred_and_plot_image(model=model,
                    image_path=custom_image_path,
                    class_names=predict.class_names,
                    device=predict.device)
