from PIL import Image
import os

# Input folder containing your images
input_folder = '../diabetic-retinopathy-detection/test'

# Output folder where resized images will be saved
output_folder = '../diabetic-retinopathy-detection/resized_test_650'

# Desired resolution
new_width = 650
new_height = 650

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all files in the input folder
files = os.listdir(input_folder)

for filename in files:
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open and resize the image
        img = Image.open(input_path)
        img = img.resize((new_width, new_height), Image.ANTIALIAS)

        # Save the resized image
        img.save(output_path)

        print(f'Resized {filename} to {new_width}x{new_height} and saved as {output_path}')