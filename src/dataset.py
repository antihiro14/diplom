"""
Модуль роботи з датасетом.

Містить клас для завантаження, підготовки
та обробки зображень страв перед передачею у модель.
"""

import os
import uuid

from PIL import Image
from torch.utils.data import Dataset

from src.logger import setup_logger


logger = setup_logger()


class UECFoodDataset(Dataset):
    """
    Клас для роботи з датасетом зображень їжі.

    Забезпечує доступ до зображень та відповідних міток класів,
    а також виконує необхідні перетворення даних.
    """

    def __init__(self, root_dir, transform=None):
        """
        Ініціалізує датасет.

        Args:
            root_dir: Шлях до директорії з даними.
            transform: Перетворення, які застосовуються до зображень.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        logger.info(f"Ініціалізація датасету. Шлях: {root_dir}")

        try:
            if not os.path.exists(root_dir):
                raise FileNotFoundError(f"Директорію не знайдено: {root_dir}")

            for label in os.listdir(root_dir):
                class_dir = os.path.join(root_dir, label)

                if not os.path.isdir(class_dir):
                    logger.debug(f"Пропущено не директорію: {class_dir}")
                    continue

                try:
                    label_int = int(label) - 1
                except ValueError:
                    logger.warning(f"Пропущено директорію з некоректною міткою: {label}")
                    continue

                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)

                    if img_path.endswith(".jpg"):
                        self.samples.append((img_path, label_int))

            logger.info(f"Датасет успішно ініціалізовано. Кількість зразків: {len(self.samples)}")

        except FileNotFoundError as e:
            error_id = uuid.uuid4()
            logger.error(
                f"[ERROR_ID={error_id}] Не знайдено директорію датасету: {e}",
                exc_info=True
            )
            raise

        except Exception as e:
            error_id = uuid.uuid4()
            logger.error(
                f"[ERROR_ID={error_id}] Помилка під час ініціалізації датасету: {e}",
                exc_info=True
            )
            raise

    def __len__(self):
        """
        Повертає кількість елементів датасету.

        Returns:
            Кількість доступних зразків.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Повертає елемент датасету за індексом.

        Args:
            idx: Індекс елемента.

        Returns:
            Кортеж (зображення, мітка класу).
        """
        try:
            img_path, label = self.samples[idx]

            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, label

        except IndexError as e:
            error_id = uuid.uuid4()
            logger.error(
                f"[ERROR_ID={error_id}] Невірний індекс датасету: {idx}. {e}",
                exc_info=True
            )
            raise

        except FileNotFoundError as e:
            error_id = uuid.uuid4()
            logger.error(
                f"[ERROR_ID={error_id}] Не знайдено файл зображення: {e}",
                exc_info=True
            )
            raise

        except Exception as e:
            error_id = uuid.uuid4()
            logger.error(
                f"[ERROR_ID={error_id}] Помилка при отриманні елемента датасету: {e}",
                exc_info=True
            )
            raise