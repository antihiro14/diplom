import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.dataset import UECFoodDataset
from models.classifier import get_model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = UECFoodDataset("data/UECFOOD256", transform=transform)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_classes = 256
    model = get_model(num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "models/model.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()