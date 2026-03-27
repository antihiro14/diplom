import torchvision.transforms as transforms
from src.dataset import UECFoodDataset

dataset = UECFoodDataset(
    root_dir="data/UECFOOD256",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
)

print("Dataset size:", len(dataset))

img, label = dataset[0]
print("Image shape:", img.shape)
print("Label:", label)