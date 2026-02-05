import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

# Class mapping for multiclass deforestation detection (from build_dataset.py)
CLASS_NAMES = {
    0: "forest",
    1: "desmatamento_cr",        # Desmatamento corte raso
    2: "desmatamento_veg",       # Desmatamento com vegetação
    3: "mineracao",              # Mineração
    4: "degradacao",             # Degradação
    5: "cicatriz_de_queimada",   # Cicatriz de incêndio florestal
    6: "cs_desordenado",         # Corte seletivo Desordenado
    7: "cs_geometrico",          # Corte seletivo Geométrico
    8: "nao_definido",           # Non-defined deforestation
}

# Reverse mapping: class name -> label ID
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}

class SegmentClassificationDataset(Dataset):
    def __init__(self, root_dir, regions, transform=False, return_dict=True, exc_labels=None):
        self.root_dir = root_dir
        self.regions = regions
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.return_dict = return_dict
        
        # Populate image paths and labels based on specified regions
        self._load_images()

        # Remove examples with labels in exc_labels
        if exc_labels:
            self._remove_examples(exc_labels)
    
    def _remove_examples(self, exc_labels):
        """Helper function to remove examples with labels in exc_labels."""
        new_image_paths = []
        new_labels = []
        for i in range(len(self.image_paths)):
            if self.labels[i] not in exc_labels:
                new_image_paths.append(self.image_paths[i])
                new_labels.append(self.labels[i])
        self.image_paths = new_image_paths
        self.labels = new_labels

    def _load_images(self):
        """Helper function to load image paths and labels based on regions and transform setting."""
        for region in self.regions:
            original_dir = os.path.join(self.root_dir, region, 'original')
            augmented_dir = os.path.join(self.root_dir, region, 'augmented')
            
            # Load original images
            self._load_images_from_folder(original_dir)
            
            # Load augmented images if transform is provided
            if self.transform:
                self._load_images_from_folder(augmented_dir)

    def _load_images_from_folder(self, folder):
        """Helper function to load images and labels from a specific folder."""
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                if filename.endswith('.png'):
                    # Get full path to image
                    file_path = os.path.join(folder, filename)
                    self.image_paths.append(file_path)
                    
                    # Extract class name from filename (format: {id}_{class_name}.png or {id}_{class_name}_aug_{i}.png)
                    parts = filename.replace('.png', '').split('_')
                    # Handle augmented files: {id}_{class_name}_aug_{i}
                    if 'aug' in parts:
                        aug_idx = parts.index('aug')
                        class_name = '_'.join(parts[1:aug_idx])
                    else:
                        class_name = '_'.join(parts[1:])
                    
                    # Map class name to label ID
                    label = CLASS_NAME_TO_ID.get(class_name, -1)
                    if label == -1:
                        print(f"Warning: Unknown class '{class_name}' in file {filename}")
                        continue
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        # Apply ToTensor and ImageNet normalization (required for pretrained models)
        mandatory_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = mandatory_transforms(image)
        label = int(label)

        image = image.float()
        label = torch.tensor(label, dtype=torch.long)

        if self.return_dict:
            return {'image': image, 'label': label}
        
        return image, label
