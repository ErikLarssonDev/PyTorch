import torch 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4 
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 64
NUM_EPOCHS = 100 
NUM_WORKERS = 6 # 0
CHECKPOINT_FILE = "b3.pth.tar"
PIN_MEMORY = True
SAVE_MODEL = True 
LOAD_MODEL = True 
IMAGES_FOLDER = "../train/images_resized_650/" # TODO: Check if this needs changing when the dataset has downloaded
PATH_TO_CSV_TRAIN = "../train/train_labels.csv"
PATH_TO_CSV_VAL = "../train/val_labels.csv"
IMAGES_FOLDER_TEST = "../test/images_resized_650/"
PATH_TO_CSV_TEST = "../train/test_labels.csv"

# Data augmentation for images
train_transforms = A.Compose(
    A.Resize(width=150, height=150),
    A.RandomCrop(width=120, height=120),
    A.Normalize( # Calculated from training set
        mean=[0.3199, 0.2240, 0.1609],
        std=[0.3020, 0.2193, 0.1741],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
)

val_transforms = A.Compose(
    A.Resize(width=120, height=120),
    A.Normalize(
        mean=[0.3199, 0.2240, 0.1609],
        std=[0.3020, 0.2193, 0.1741],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
)