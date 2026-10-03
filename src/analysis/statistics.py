# Стандартная библиотека Python
from pathlib import Path

# Сторонние библиотеки для компьютерного зрения, анализа данных и визуализации
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Локальные модули проекта
from src.config import Config as ProjectConfig


def collect_dataset_statistics(split):
    """
    Собирает статистику изображений и индексных масок выборки.

    Args:
        split (str):
            Название выборки: train, val или test.

    Returns:
        pd.DataFrame:
            Таблица статистики изображений и классов.
    """
    images_dir = ProjectConfig.RAW_IMAGES_ROOT / split
    labels_dir = ProjectConfig.INDEXED_LABELS_ROOT / split

    image_paths = sorted(images_dir.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(
            f"В директории {images_dir} не найдены изображения."
        )

    records = []

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.png"

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Не удалось прочитать изображение: {image_path}"
            )

        mask = cv2.imread(
            str(label_path),
            cv2.IMREAD_UNCHANGED
        )

        if mask is None:
            raise ValueError(
                f"Не удалось прочитать маску: {label_path}"
            )

        if mask.ndim != 2:
            raise ValueError(
                f"Маска {label_path.name} должна быть одноканальной, "
                f"получена форма {mask.shape}."
            )

        unique_values = np.unique(mask)

        if not np.isin(
            unique_values,
            list(ProjectConfig.CLASS_IDS.values())
        ).all():
            raise ValueError(
                f"Маска {label_path.name} содержит недопустимые классы: "
                f"{unique_values}."
            )

        height, width = mask.shape
        image_area = height * width

        pixel_counts = {
            class_id: int(np.sum(mask == class_id))
            for class_id in ProjectConfig.CLASS_IDS.values()
        }

        class_area_ratios = {
            class_id: pixel_counts[class_id] / image_area
            for class_id in ProjectConfig.CLASS_IDS.values()
        }

        object_areas = {}

        for class_name in ("cat", "dog"):
            class_id = ProjectConfig.CLASS_IDS[class_name]
            binary_mask = (mask == class_id).astype(np.uint8)

            _, _, component_stats, _ = cv2.connectedComponentsWithStats(
                binary_mask,
                connectivity=8
            )

            areas = component_stats[
                1:,
                cv2.CC_STAT_AREA
            ].tolist()

            object_areas[class_name] = areas

        records.append(
            {
                "split": split,
                "image_name": image_path.name,
                "width": width,
                "height": height,
                "image_area": image_area,
                "aspect_ratio": width / height,
                "background_pixels": pixel_counts[
                    ProjectConfig.CLASS_IDS["background"]
                ],
                "cat_pixels": pixel_counts[
                    ProjectConfig.CLASS_IDS["cat"]
                ],
                "dog_pixels": pixel_counts[
                    ProjectConfig.CLASS_IDS["dog"]
                ],
                "background_area_ratio": class_area_ratios[
                    ProjectConfig.CLASS_IDS["background"]
                ],
                "cat_area_ratio": class_area_ratios[
                    ProjectConfig.CLASS_IDS["cat"]
                ],
                "dog_area_ratio": class_area_ratios[
                    ProjectConfig.CLASS_IDS["dog"]
                ],
                "cat_object_count": len(object_areas["cat"]),
                "dog_object_count": len(object_areas["dog"]),
                "cat_object_areas": object_areas["cat"],
                "dog_object_areas": object_areas["dog"],
            }
        )

    statistics = pd.DataFrame(records)

    print("=" * 60)
    print("СТАТИСТИКА ДАТАСЕТА")
    print("=" * 60)
    print(f"Выборка: {split}")
    print(f"Изображений обработано: {len(statistics)}")
    print(f"Источник изображений: {images_dir}")
    print(f"Источник индексных масок: {labels_dir}")
    print("=" * 60)

    return statistics


def summarize_dataset_statistics(statistics):
    """
    Выводит сводную статистику датасета.

    Args:
        statistics (pandas.DataFrame):
            Таблица со статистикой изображений.

    Returns:
        None.
    """
    if statistics.empty:
        print("Статистика пуста.")
        return

    split_counts = statistics.groupby("split").size()

    print("=" * 60)
    print("СВОДНАЯ СТАТИСТИКА ДАТАСЕТА")
    print("=" * 60)

    print("\nКоличество изображений:")

    for split, count in split_counts.items():
        print(f"  {split}: {count}")

    print("\nРазмеры изображений:")

    print(
        f"  Ширина: "
        f"min={statistics['width'].min()}, "
        f"max={statistics['width'].max()}, "
        f"mean={statistics['width'].mean():.1f}"
    )

    print(
        f"  Высота: "
        f"min={statistics['height'].min()}, "
        f"max={statistics['height'].max()}, "
        f"mean={statistics['height'].mean():.1f}"
    )

    print(
        f"  Aspect ratio: "
        f"min={statistics['aspect_ratio'].min():.2f}, "
        f"max={statistics['aspect_ratio'].max():.2f}, "
        f"mean={statistics['aspect_ratio'].mean():.2f}"
    )

    print("\nСреднее количество объектов на изображении:")

    print(
        f"  Кошки: "
        f"{statistics['cat_object_count'].mean():.2f}"
    )

    print(
        f"  Собаки: "
        f"{statistics['dog_object_count'].mean():.2f}"
    )

    print("\nСредняя доля изображения, занятая объектами:")

    print(
        f"  Кошки: "
        f"{statistics['cat_area_ratio'].mean() * 100:.2f}%"
    )

    print(
        f"  Собаки: "
        f"{statistics['dog_area_ratio'].mean() * 100:.2f}%"
    )

    print("=" * 60)


def calculate_class_balance(statistics):
    """
    Рассчитывает баланс классов по количеству пикселей.

    Args:
        statistics (pandas.DataFrame):
            Таблица со статистикой изображений.

    Returns:
        pandas.DataFrame:
            Таблица с количеством и долей пикселей каждого класса.
    """
    pixel_columns = {
        "background": "background_pixels",
        "cat": "cat_pixels",
        "dog": "dog_pixels"
    }

    class_pixels = {
        class_name: int(statistics[column].sum())
        for class_name, column in pixel_columns.items()
    }

    total_pixels = sum(class_pixels.values())
    records = []

    for class_name, pixel_count in class_pixels.items():
        records.append(
            {
                "class": class_name,
                "pixels": pixel_count,
                "percentage": pixel_count / total_pixels * 100
            }
        )

    class_balance = pd.DataFrame(records)

    print("=" * 60)
    print("БАЛАНС КЛАССОВ")
    print("=" * 60)
    print(f"Всего пикселей: {total_pixels}")

    for row in class_balance.itertuples(index=False):
        print(
            f"  {row[0]}: "
            f"{row[1]:,} пикселей "
            f"({row[2]:.2f}%)"
        )

    print("=" * 60)

    return class_balance


def plot_class_balance(class_balance):
    """
    Строит график распределения классов по пикселям.

    Args:
        class_balance (pandas.DataFrame):
            Таблица с балансом классов.

    Returns:
        None.
    """
    print("=" * 60)
    print("ГРАФИК БАЛАНСА КЛАССОВ")
    print("=" * 60)

    plt.figure(figsize=(9, 6))

    plt.bar(
        class_balance["class"],
        class_balance["percentage"]
    )

    plt.xlabel("Класс")
    plt.ylabel("Доля пикселей, %")
    plt.title("Распределение классов по пикселям")
    plt.grid(axis="y", alpha=0.3)
    plt.show()

    print("График баланса классов построен.")


def plot_image_size_distributions(statistics):
    """
    Строит графики распределения размеров изображений.

    Args:
        statistics (pandas.DataFrame):
            Таблица со статистикой изображений.

    Returns:
        None.
    """
    print("=" * 60)
    print("РАСПРЕДЕЛЕНИЕ РАЗМЕРОВ ИЗОБРАЖЕНИЙ")
    print("=" * 60)

    print("\nШирина изображений:")

    print(
        f"  Минимум: {statistics['width'].min()} пикселей"
    )

    print(
        f"  Максимум: {statistics['width'].max()} пикселей"
    )

    print(
        f"  Среднее: {statistics['width'].mean():.1f} пикселей"
    )

    print(
        f"  Медиана: {statistics['width'].median():.1f} пикселей"
    )

    print("\nВысота изображений:")

    print(
        f"  Минимум: {statistics['height'].min()} пикселей"
    )

    print(
        f"  Максимум: {statistics['height'].max()} пикселей"
    )

    print(
        f"  Среднее: {statistics['height'].mean():.1f} пикселей"
    )

    print(
        f"  Медиана: {statistics['height'].median():.1f} пикселей"
    )

    print("\nСоотношение сторон (aspect ratio):")

    print(
        f"  Минимум: "
        f"{statistics['aspect_ratio'].min():.2f}"
    )

    print(
        f"  Максимум: "
        f"{statistics['aspect_ratio'].max():.2f}"
    )

    print(
        f"  Среднее: "
        f"{statistics['aspect_ratio'].mean():.2f}"
    )

    print(
        f"  Медиана: "
        f"{statistics['aspect_ratio'].median():.2f}"
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        statistics["width"],
        bins=20
    )

    plt.xlabel("Ширина, пиксели")
    plt.ylabel("Количество изображений")
    plt.title("Распределение ширины изображений")
    plt.grid(axis="y", alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))

    plt.hist(
        statistics["height"],
        bins=20
    )

    plt.xlabel("Высота, пиксели")
    plt.ylabel("Количество изображений")
    plt.title("Распределение высоты изображений")
    plt.grid(axis="y", alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))

    plt.scatter(
        statistics["width"],
        statistics["height"],
        alpha=0.6
    )

    plt.xlabel("Ширина, пиксели")
    plt.ylabel("Высота, пиксели")
    plt.title("Соотношение ширины и высоты изображений")
    plt.grid(alpha=0.3)
    plt.show()

    print("Графики размеров изображений построены.")


def collect_object_statistics(statistics):
    """
    Преобразует площади объектов в отдельную таблицу.

    Args:
        statistics (pandas.DataFrame):
            Таблица со статистикой изображений.

    Returns:
        pandas.DataFrame:
            Таблица площадей отдельных объектов.
    """
    records = []

    for row in statistics.itertuples(index=False):
        for class_name in ("cat", "dog"):
            object_areas = (
                row.cat_object_areas
                if class_name == "cat"
                else row.dog_object_areas
            )

            for object_number, area in enumerate(
                object_areas,
                start=1
            ):
                records.append(
                    {
                        "split": row.split,
                        "image_name": row.image_name,
                        "class": class_name,
                        "object_number": object_number,
                        "area_pixels": area,
                        "area_ratio": area / row.image_area
                    }
                )

    object_statistics = pd.DataFrame(records)

    print("=" * 60)
    print("СТАТИСТИКА РАЗМЕРОВ ОБЪЕКТОВ")
    print("=" * 60)
    print(f"Всего объектов: {len(object_statistics)}")

    if not object_statistics.empty:
        for class_name in ("cat", "dog"):
            class_objects = object_statistics[
                object_statistics["class"] == class_name
            ]

            print(f"\nКласс: {class_name}")

            print(
                f"  Объектов: "
                f"{len(class_objects)}"
            )

            print(
                f"  Площадь: "
                f"min={class_objects['area_pixels'].min():.0f}, "
                f"max={class_objects['area_pixels'].max():.0f}, "
                f"mean={class_objects['area_pixels'].mean():.0f}"
            )

            print(
                f"  Доля изображения: "
                f"mean={class_objects['area_ratio'].mean() * 100:.2f}%"
            )

    print("=" * 60)

    return object_statistics


def find_small_objects(
    object_statistics,
    class_name="dog",
    max_area=1
):
    """
    Находит изображения с очень маленькими объектами заданного класса.

    Args:
        object_statistics (pandas.DataFrame):
            Таблица со статистикой отдельных объектов.

        class_name (str):
            Название класса для поиска.

        max_area (int):
            Максимальная площадь объекта в пикселях.

    Returns:
        pandas.DataFrame:
            Таблица найденных маленьких объектов.
    """
    small_objects = object_statistics[
        (object_statistics["class"] == class_name)
        & (object_statistics["area_pixels"] <= max_area)
    ].copy()

    print("=" * 60)
    print("ПОИСК МАЛЕНЬКИХ ОБЪЕКТОВ")
    print("=" * 60)
    print(f"Класс: {class_name}")
    print(f"Максимальная площадь: {max_area} пикселей")
    print(f"Найдено объектов: {len(small_objects)}")

    if small_objects.empty:
        print("Маленькие объекты не найдены.")
        print("=" * 60)
        return small_objects

    print("\nНайденные объекты:")

    for row in small_objects.itertuples(index=False):
        print(
            f"  Датасет: {row.split}, "
            f"изображение: {row.image_name}, "
            f"площадь={row.area_pixels} пиксель, "
            f"номер объекта={row.object_number}"
        )

    print("=" * 60)

    return small_objects


def plot_object_size_distributions(object_statistics):
    """
    Строит графики распределения размеров объектов.

    Args:
        object_statistics (pandas.DataFrame):
            Таблица площадей отдельных объектов.

    Returns:
        None.
    """
    print("=" * 60)
    print("РАСПРЕДЕЛЕНИЕ РАЗМЕРОВ ОБЪЕКТОВ")
    print("=" * 60)

    if object_statistics.empty:
        print("Объекты для построения графиков не найдены.")
        return

    for class_name in ("cat", "dog"):
        class_objects = object_statistics[
            object_statistics["class"] == class_name
        ]

        plt.figure(figsize=(10, 6))

        plt.hist(
            class_objects["area_ratio"] * 100,
            bins=20
        )

        plt.xlabel("Доля изображения, %")
        plt.ylabel("Количество объектов")
        plt.title(
            "Распределение относительных размеров объектов: "
            f"{class_name}"
        )
        plt.grid(axis="y", alpha=0.3)
        plt.show()

    plt.figure(figsize=(10, 6))

    for class_name in ("cat", "dog"):
        class_objects = object_statistics[
            object_statistics["class"] == class_name
        ]

        plt.scatter(
            class_objects["area_pixels"],
            class_objects["area_ratio"] * 100,
            alpha=0.6,
            label=class_name
        )

    plt.xlabel("Площадь объекта, пиксели")
    plt.ylabel("Доля изображения, %")
    plt.title("Размеры объектов")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print("Графики размеров объектов построены.")


def check_dataset_quality(split):
    """
    Проверяет качество изображений и индексных масок выборки.

    Args:
        split (str):
            Название выборки: train, val или test.

    Returns:
        dict:
            Результаты проверки качества данных.
    """
    images_dir = ProjectConfig.RAW_IMAGES_ROOT / split
    labels_dir = ProjectConfig.INDEXED_LABELS_ROOT / split

    image_paths = sorted(images_dir.glob("*.jpg"))

    total_images = len(image_paths)
    valid_images = 0
    invalid_images = 0
    invalid_masks = 0
    missing_masks = 0

    allowed_class_ids = set(
        ProjectConfig.CLASS_IDS.values()
    )

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.png"

        image = cv2.imread(str(image_path))
        mask = cv2.imread(
            str(label_path),
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            invalid_images += 1
            continue

        valid_images += 1

        if mask is None:
            missing_masks += 1
            continue

        if mask.ndim != 2:
            invalid_masks += 1
            continue

        unique_values = set(
            np.unique(mask).tolist()
        )

        if not unique_values.issubset(
            allowed_class_ids
        ):
            invalid_masks += 1

    result = {
        "split": split,
        "total_images": total_images,
        "valid_images": valid_images,
        "invalid_images": invalid_images,
        "missing_masks": missing_masks,
        "invalid_masks": invalid_masks,
    }

    print("=" * 60)
    print("ПРОВЕРКА КАЧЕСТВА ДАННЫХ")
    print("=" * 60)
    print(f"Выборка: {split}")
    print(f"Всего изображений: {total_images}")
    print(f"Корректных изображений: {valid_images}")
    print(f"Некорректных изображений: {invalid_images}")
    print(f"Отсутствующих масок: {missing_masks}")
    print(f"Некорректных индексных масок: {invalid_masks}")
    print(f"Источник индексных масок: {labels_dir}")
    print("=" * 60)

    return result


def show_eda_examples(split, image_names):
    """
    Показывает изображения с наложенными цветными сегментационными масками
    в одну горизонтальную строку.

    Args:
        split (str):
            Название выборки: train, val или test.

        image_names (list):
            Список имён изображений для отображения.

    Returns:
        None.
    """
    images_dir = ProjectConfig.RAW_IMAGES_ROOT / split
    labels_dir = ProjectConfig.INDEXED_LABELS_ROOT / split

    class_colors = {
        ProjectConfig.CLASS_IDS["cat"]: (255, 120, 0),
        ProjectConfig.CLASS_IDS["dog"]: (120, 80, 255),
    }

    alpha = 0.45
    overlays = []
    valid_image_names = []

    print("=" * 60)
    print("ВИЗУАЛЬНЫЙ АНАЛИЗ ДАТАСЕТА")
    print("=" * 60)
    print(f"Выборка: {split}")
    print(f"Изображений запрошено: {len(image_names)}")

    for image_name in image_names:
        image_path = images_dir / image_name
        label_path = labels_dir / f"{Path(image_name).stem}.png"

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            print(
                f"Предупреждение: не удалось прочитать "
                f"изображение {image_path}"
            )
            continue

        if mask is None:
            print(
                f"Предупреждение: не удалось прочитать "
                f"маску {label_path}"
            )
            continue

        if mask.ndim != 2:
            raise ValueError(
                f"Маска {label_path.name} должна быть одноканальной, "
                f"получена форма {mask.shape}."
            )

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Размеры изображения и маски не совпадают: "
                f"{image_name}, "
                f"изображение={image.shape[:2]}, "
                f"маска={mask.shape[:2]}."
            )

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay = image_rgb.copy()

        for class_id, color in class_colors.items():
            class_mask = mask == class_id

            if not np.any(class_mask):
                continue

            color_array = np.array(color, dtype=np.uint8)

            overlay[class_mask] = (
                (1 - alpha) * overlay[class_mask]
                + alpha * color_array
            ).astype(np.uint8)

            contours, _ = cv2.findContours(
                class_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                overlay,
                contours,
                -1,
                color,
                2
            )

        overlays.append(overlay)
        valid_image_names.append(image_name)

    if not overlays:
        print("Нет изображений для отображения.")
        return

    figure, axes = plt.subplots(
        1,
        len(overlays),
        figsize=(30, 4)
    )

    if len(overlays) == 1:
        axes = [axes]

    for axis, overlay, image_name in zip(
        axes,
        overlays,
        valid_image_names
    ):
        axis.imshow(overlay)
        axis.set_title(image_name, fontsize=8)
        axis.axis("off")

    figure.suptitle(
        f"{split}: изображения с наложенными сегментационными масками",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

    print(f"Изображений отображено: {len(overlays)}")
    print("Отображение завершено.")