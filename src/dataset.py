import os
from PIL import Image
from torch.utils.data import Dataset

class UECFoodDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for label in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, label)

            if not os.path.isdir(class_dir):
                continue

            # label -> int
            label_int = int(label) - 1  # делаем с 0

            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)

                if img_path.endswith(".jpg"):
                    self.samples.append((img_path, label_int))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label