import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

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
                    
                    label = filename.split('_')[1].split('.')[0]
                    label = 0 if label == 'forest' else 1 if 'recent-def' in label else 2
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        # Apply ToTensor and Normalize transforms independently
        mandatory_transforms = T.Compose([
            T.ToTensor(),
        ])

        image = mandatory_transforms(image)
        label = int(label)

        image = image.float()
        label = torch.tensor(label, dtype=torch.long)

        if self.return_dict:
            return {'image': image, 'label': label}
        
        return image, label
