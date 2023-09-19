import os
import random
import shutil

# Define the paths to your original data folders
data_folder = '../carvana-image-masking-challenge'  # Update with your actual data folder
train_images_folder = os.path.join(data_folder, 'train')
train_masks_folder = os.path.join(data_folder, 'train_masks')
output_folder = 'data'  # Create a new folder for the output data

# Create output folders if they don't exist
os.makedirs(os.path.join(output_folder, 'train_images'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'train_masks'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'val_images'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'val_masks'), exist_ok=True)

# List all the image files in the train_images folder
image_files = os.listdir(train_images_folder)

# Shuffle the list of image files to randomize the split
random.shuffle(image_files)

# Calculate the split index based on your desired split ratio
split_index = int(0.85 * len(image_files))

# Split the data into training and validation sets
train_image_files = image_files[:split_index]
val_image_files = image_files[split_index:]

# Copy the corresponding images and masks to the output folders
for image_file in train_image_files:
    image_path = os.path.join(train_images_folder, image_file)
    mask_file = image_file.replace(".jpg", "_mask.gif")
    mask_path = os.path.join(train_masks_folder, mask_file)

    # Copy the image and mask to the training folders
    shutil.copy(image_path, os.path.join(output_folder, 'train_images'))
    shutil.copy(mask_path, os.path.join(output_folder, 'train_masks'))

for image_file in val_image_files:
    image_path = os.path.join(train_images_folder, image_file)
    mask_file = image_file.replace(".jpg", "_mask.gif")
    mask_path = os.path.join(train_masks_folder, mask_file)

    # Copy the image and mask to the validation folders
    shutil.copy(image_path, os.path.join(output_folder, 'val_images'))
    shutil.copy(mask_path, os.path.join(output_folder, 'val_masks'))

print("Data split completed successfully.")
