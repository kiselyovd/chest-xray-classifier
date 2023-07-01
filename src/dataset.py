from pathlib import Path

import numpy as np

import torch
from torchvision.io.image import read_image
from torchvision import transforms

from .preprocessing import augmentator


class ChestXRayDataset(torch.utils.data.Dataset):
    """Класс загрузки и предобработки датасета"""

    def __init__(self,
                 path_array: list[str | Path],
                 image_preprocess: None | transforms.Compose = None,
                 augmented: bool = False,
                 device: str | torch.device = "cpu"
                 ):
        """
        Инициализация класса датасета
        
        :param path_array: список путей до изображений
        :param image_preprocess: функция предобработки изображения
        :param augmented: аугментирование данных
        :param device: устройство, на котором будут происходить все вычисления
        """
        self.path_array = path_array
        self.image_preprocess = image_preprocess
        self.augmented = augmented
        self.device = device

    def __len__(self):
        """Метод возвращения длины датасета"""
        return len(self.path_array)

    def __getitem__(self, idx: int):
        """
        Магический метод возвращения элемента по индексу
        
        :param idx: индекс списка ссылок на изображение
        
        :return: tensor(изображение), int(метка данных)
        """
        # Получение пути
        image_path = self.path_array[idx]
        image_path = Path(image_path) if isinstance(image_path, str) else image_path

        # Чтение изображения
        image_raw = read_image(str(image_path)).to(self.device)

        if image_raw.shape[0] < 3:
            # Создание пустых тензоров для двух каналов
            empty_channels = torch.zeros(3 - image_raw.shape[0], image_raw.shape[1], image_raw.shape[2]).to(self.device)
            # Объединение исходного изображения и пустых тензоров по оси 0
            image_raw = torch.cat([image_raw, empty_channels], dim=0)

        # Определение класса
        label = 0
        if "virus" in image_path.stem:
            label = 1
        elif "bacteria" in image_path.stem:
            label = 2
        # label = np.array([label], dtype=np.int64)
            
        # Предобработка и аугментация
        if self.image_preprocess:
            image = self.image_preprocess(image_raw)
            if self.augmented:
                image = augmentator(image).to(torch.float)
        else:
            image = transforms.ToTensor()(image_raw)

        return image, label
