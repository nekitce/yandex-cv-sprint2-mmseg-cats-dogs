# Стандартная библиотека Python
import os
import random
import re

# Сторонние библиотеки для работы с данными
import numpy as np
import pandas as pd

# PyTorch
import torch


def set_seed(seed=42, verbose=False):
    """
    Фиксирует генераторы случайных чисел.

    Args:
        seed (int):
            Значение seed.

        verbose (bool):
            Выводить сообщение об успешной установке.

    Returns:
        None.
    """

    # Фиксация генераторов Python и NumPy
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Фиксация генераторов PyTorch
    torch.manual_seed(seed)

    # Фиксация генераторов CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Настройка воспроизводимого поведения CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if verbose:
        print(f"Seed {seed} успешно установлен!")