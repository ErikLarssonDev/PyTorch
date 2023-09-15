# import config
import numpy as np
import os
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from utils import (iou_width_height as iou,
                   non_max_suppression as nms)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLODataset(Dataset):
    def __init__(
            self,
            csv_file,
            img_dir,
            label_dir,
            anchors,
            image_size=416,
            S=[13, 26, 52],
            C=20,
            transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) # Putting all anchors of different scales together
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3 # We are assuming that we have 3 different scales but we can alter the number of anchors
        self.C = C 
        self.ignore_iou_thresh = 0.5 # One anchor responsible for each cell, the responsible one will be the one with the highest iou to the ground truth, discards multiple predictions with iou > 0.5
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist() # Using np.roll to get classes at the end
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB")) # Make sure it is in RGB, think there is greyscale in COCO

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S] # [p_o, x, y, w, h, c] 
        
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box 
            has_anchor = [False, False, False]

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale # 0, 1, 2, which scale
                anchor_on_scale = anchor_idx & self.num_anchors_per_scale # 0, 1, 2, which anchor
                S = self.S[scale_idx]
                i, j = int(S*y), int(S*x) # x=0.5, S=13 --> int(6.5) = 6
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S*x - j, S*y - i # both are between [0, 1]
                    width_cell, height_cell = width*S, height*S 
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    # Set anchors of scale_idx to true
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore this prediction 
                    
        return image, tuple(targets)