import torch 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
NUM_WORKERS = 0
BATCH_SIZE = 64
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_FILE = "b7.pth.tar"
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1

basic_transform = A.Compose(
    [
        A.Resize(height=448, width=448),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)