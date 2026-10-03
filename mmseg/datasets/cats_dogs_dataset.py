# Сторонние библиотеки для работы с MMSegmentation
from mmseg.registry import DATASETS

# Базовый класс датасета MMSegmentation
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CatsDogsDataset(BaseSegDataset):
    """
    Датасет для многоклассовой сегментации кошек и собак.

    Args:
        *args:
            Позиционные аргументы базового класса BaseSegDataset.

        **kwargs:
            Именованные аргументы базового класса BaseSegDataset.

    Returns:
        None.
    """

    METAINFO = dict(
        classes=("background", "cat", "dog"),
        palette=[
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
        ],
    )

    def __init__(self, *args, **kwargs):
        """
        Инициализирует датасет кошек и собак.

        Args:
            *args:
                Позиционные аргументы базового класса BaseSegDataset.

            **kwargs:
                Именованные аргументы базового класса BaseSegDataset.

        Returns:
            None.
        """
        super().__init__(*args, **kwargs)

        print("=" * 60)
        print("ИНИЦИАЛИЗАЦИЯ CATSDOGS DATASET")
        print("=" * 60)
        print(f"Количество изображений: {len(self)}")
        print(f"Классы: {self.METAINFO['classes']}")
        print("=" * 60)