"""
Головний модуль запуску проєкту.

Цей файл є точкою входу в систему аналізу їжі за фотографією.
На поточному етапі використовується для перевірки завантаження
датасету та базової роботи зображень і міток.
"""

import uuid

import torchvision.transforms as transforms

from src.dataset import UECFoodDataset
from src.logger import setup_logger


logger = setup_logger()


def main():
    """
    Основна функція запуску програми.

    Ініціалізує датасет, виконує базову перевірку його доступності
    та виводить інформацію про перший елемент.

    Returns:
        None
    """
    logger.info("Програма запущена")

    try:
        logger.info("Початок ініціалізації датасету")

        dataset = UECFoodDataset(
            root_dir="data/UECFOOD256",
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        )

        dataset_size = len(dataset)
        logger.info(f"Датасет успішно завантажено. Кількість елементів: {dataset_size}")
        print("Dataset size:", dataset_size)

        if dataset_size == 0:
            logger.warning("Датасет порожній")
            print("Dataset is empty")
            return

        logger.info("Отримання першого елемента датасету")
        img, label = dataset[0]

        logger.info(f"Перший елемент успішно отримано. Label: {label}")
        print("Image shape:", img.shape)
        print("Label:", label)

    except FileNotFoundError as e:
        error_id = uuid.uuid4()
        logger.error(
            f"[ERROR_ID={error_id}] Не знайдено файл або директорію датасету: {e}",
            exc_info=True
        )
        print(f"Помилка: не знайдено файл або папку. Код помилки: {error_id}")

    except IndexError as e:
        error_id = uuid.uuid4()
        logger.error(
            f"[ERROR_ID={error_id}] Помилка доступу до елемента датасету: {e}",
            exc_info=True
        )
        print(f"Помилка: не вдалося отримати елемент датасету. Код помилки: {error_id}")

    except Exception as e:
        error_id = uuid.uuid4()
        logger.error(
            f"[ERROR_ID={error_id}] Невідома помилка під час виконання програми: {e}",
            exc_info=True
        )
        print(f"Сталася неочікувана помилка. Код помилки: {error_id}")

    finally:
        logger.info("Програма завершена")


if __name__ == "__main__":
    main()