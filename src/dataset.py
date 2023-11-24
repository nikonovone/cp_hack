from pathlib import Path

import cv2
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, folder, augmentations=None):
        folder = Path(folder)

        self.label_map = {
            "Бетон": 0,
            "Грунт": 1,
            "Дерево": 2,
            "Кирпич": 3,
        }
        self.images = sorted(list(folder.rglob("*.jpg")))
        self.augmentations = augmentations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.label_map[image_path.parent.name]
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"] / 255
            return image, label
        else:
            return image / 255, label
