"""
Модуль навчання моделі.

Містить логіку тренування нейронної мережі для класифікації
зображень страв.
"""

import uuid

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.dataset import UECFoodDataset
from models.classifier import get_model
from src.logger import setup_logger


logger = setup_logger()


def train():
    """
    Виконує навчання моделі класифікації.

    Основні етапи:
    - ініціалізація датасету
    - створення DataLoader
    - побудова моделі
    - навчання
    - збереження моделі

    Returns:
        None
    """

    logger.info("Запуск навчання моделі")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Використовується пристрій: {device}")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        logger.info("Завантаження датасету")
        dataset = UECFoodDataset("data/UECFOOD256", transform=transform)

        if len(dataset) == 0:
            logger.warning("Датасет порожній")
            return

        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        logger.info(f"DataLoader створено. Кількість батчів: {len(train_loader)}")

        num_classes = 256
        model = get_model(num_classes).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        epochs = 5
        logger.info(f"Початок навчання. Кількість епох: {epochs}")

        for epoch in range(epochs):
            model.train()
            total_loss = 0

            logger.info(f"Epoch {epoch + 1} розпочато")

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.info(f"Epoch {epoch + 1} завершено. Loss: {total_loss:.4f}")
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

        torch.save(model.state_dict(), "models/model.pth")
        logger.info("Модель успішно збережена")

    except FileNotFoundError as e:
        error_id = uuid.uuid4()
        logger.error(
            f"[ERROR_ID={error_id}] Не знайдено файли датасету: {e}",
            exc_info=True
        )
        print(f"Помилка: не знайдено файли. Код помилки: {error_id}")

    except RuntimeError as e:
        error_id = uuid.uuid4()
        logger.error(
            f"[ERROR_ID={error_id}] Помилка під час навчання (можливо GPU/пам'ять): {e}",
            exc_info=True
        )
        print(f"Помилка під час навчання. Код: {error_id}")

    except Exception as e:
        error_id = uuid.uuid4()
        logger.error(
            f"[ERROR_ID={error_id}] Невідома помилка: {e}",
            exc_info=True
        )
        print(f"Невідома помилка. Код: {error_id}")

    finally:
        logger.info("Завершення роботи train()")


if __name__ == "__main__":
    train()