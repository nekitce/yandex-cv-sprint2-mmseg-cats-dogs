# Стандартная библиотека
from pathlib import Path

# Сторонние библиотеки
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_segmentation_predictions(
    runner,
    output_directory,
):
    """
    Получает предсказания модели на тестовом датасете и сохраняет индексированные маски.

    Args:
        runner (Runner):
            Runner с загруженным checkpoint и test DataLoader.

        output_directory (Path):
            Директория для сохранения предсказанных масок.

    Returns:
        list[Path]:
            Пути к сохранённым предсказанным маскам.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    predictions = []

    for batch in runner.test_dataloader:
        outputs = runner.model.test_step(batch)

        for data_sample in outputs:
            image_path = data_sample.metainfo["img_path"]
            image_name = Path(image_path).stem

            prediction = (
                data_sample.pred_sem_seg.data
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            prediction_path = output_directory / f"{image_name}.png"
            cv2.imwrite(str(prediction_path), prediction)
            predictions.append(prediction_path)

    print("=" * 60)
    print("СОХРАНЕНИЕ ПРЕДСКАЗАНИЙ")
    print("=" * 60)
    print(f"Обработано изображений: {len(predictions)}")
    print(f"Сохранено масок: {len(predictions)}")
    print(f"Директория: {output_directory}")
    print("=" * 60)

    return predictions


def analyze_segmentation_predictions(
    runner,
    prediction_paths,
    class_ids,
):
    """
    Сравнивает предсказанные маски с эталонными масками и рассчитывает Dice по классам.

    Args:
        runner (Runner):
            Runner с тестовым Dataset, содержащим пути к эталонным маскам.

        prediction_paths (list[Path]):
            Пути к сохранённым предсказанным маскам.

        class_ids (dict):
            Словарь с идентификаторами классов.

    Returns:
        pd.DataFrame:
            Таблица с метриками для каждого изображения.
    """
    dataset = runner.test_dataloader.dataset
    ground_truth_by_image = {}

    for dataset_index in range(len(dataset)):
        data_sample = dataset[dataset_index]["data_samples"]
        image_path = Path(data_sample.metainfo["img_path"])
        ground_truth_path = Path(data_sample.metainfo["seg_map_path"])
        ground_truth_by_image[image_path.stem] = ground_truth_path

    results = []

    for prediction_path in prediction_paths:
        image_name = prediction_path.stem

        if image_name not in ground_truth_by_image:
            raise FileNotFoundError(
                f"Эталонная маска не найдена для изображения: {image_name}"
            )

        ground_truth_path = ground_truth_by_image[image_name]

        prediction = cv2.imread(
            str(prediction_path),
            cv2.IMREAD_GRAYSCALE,
        )

        ground_truth = cv2.imread(
            str(ground_truth_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if prediction is None:
            raise FileNotFoundError(
                f"Предсказанная маска не найдена или не прочитана: {prediction_path}"
            )

        if ground_truth is None:
            raise FileNotFoundError(
                f"Эталонная маска не найдена или не прочитана: {ground_truth_path}"
            )

        if prediction.shape != ground_truth.shape:
            raise ValueError(
                f"Размеры масок не совпадают для {image_name}: "
                f"{prediction.shape} и {ground_truth.shape}"
            )

        class_dice = {}

        for class_name, class_id in class_ids.items():
            prediction_class = prediction == class_id
            ground_truth_class = ground_truth == class_id

            intersection = np.logical_and(
                prediction_class,
                ground_truth_class,
            ).sum()

            prediction_area = prediction_class.sum()
            ground_truth_area = ground_truth_class.sum()

            if ground_truth_area == 0:
                dice = 0.0
            else:
                dice = (
                    2.0
                    * intersection
                    / (prediction_area + ground_truth_area)
                )

            class_dice[class_name] = dice

        object_class_dice = [
            class_dice[class_name]
            for class_name in class_ids
            if class_name != "background"
        ]

        object_dice = np.mean(object_class_dice)

        results.append(
            {
                "image": image_name,
                "cat_dice": class_dice.get("cat"),
                "dog_dice": class_dice.get("dog"),
                "object_mDice": object_dice,
            }
        )

    results_dataframe = (
        pd.DataFrame(results)
        .sort_values(
            "object_mDice",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("=" * 60)
    print("АНАЛИЗ ИНДИВИДУАЛЬНЫХ ПРЕДСКАЗАНИЙ")
    print("=" * 60)
    print(f"Тестовых изображений: {len(dataset)}")
    print(f"Проанализировано изображений: {len(results_dataframe)}")
    print(
        f"Средний Dice объектов: "
        f"{results_dataframe['object_mDice'].mean():.4f}"
    )
    print(
        f"Лучший Dice объектов: "
        f"{results_dataframe['object_mDice'].max():.4f}"
    )
    print(
        f"Худший Dice объектов: "
        f"{results_dataframe['object_mDice'].min():.4f}"
    )
    print("=" * 60)

    return results_dataframe


def analyze_prediction_errors(
    runner,
    prediction_paths,
    class_ids,
):
    """
    Анализирует типы ошибок сегментации на уровне пикселей и классов.

    Args:
        runner (Runner):
            Runner с тестовым Dataset и Ground Truth масками.

        prediction_paths (list[Path]):
            Пути к сохранённым предсказанным маскам.

        class_ids (dict):
            Словарь с идентификаторами классов.

    Returns:
        pd.DataFrame:
            Таблица с типами ошибок и распределением классов.
    """
    dataset = runner.test_dataloader.dataset
    ground_truth_by_image = {}

    for dataset_index in range(len(dataset)):
        data_sample = dataset[dataset_index]["data_samples"]
        image_path = Path(data_sample.metainfo["img_path"])
        ground_truth_path = Path(data_sample.metainfo["seg_map_path"])
        ground_truth_by_image[image_path.stem] = ground_truth_path

    class_names = list(class_ids.keys())
    background_id = class_ids["background"]
    object_class_names = [
        class_name
        for class_name in class_names
        if class_name != "background"
    ]

    results = []

    for prediction_path in prediction_paths:
        image_name = prediction_path.stem

        if image_name not in ground_truth_by_image:
            raise FileNotFoundError(
                f"Эталонная маска не найдена для изображения: {image_name}"
            )

        ground_truth_path = ground_truth_by_image[image_name]

        prediction = cv2.imread(
            str(prediction_path),
            cv2.IMREAD_GRAYSCALE,
        )

        ground_truth = cv2.imread(
            str(ground_truth_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if prediction is None:
            raise FileNotFoundError(
                f"Предсказанная маска не найдена: {prediction_path}"
            )

        if ground_truth is None:
            raise FileNotFoundError(
                f"Ground Truth не найден: {ground_truth_path}"
            )

        if prediction.shape != ground_truth.shape:
            raise ValueError(
                f"Размеры масок не совпадают для {image_name}: "
                f"{prediction.shape} и {ground_truth.shape}"
            )

        total_pixels = prediction.size

        gt_percentages = {}
        prediction_percentages = {}

        for class_name, class_id in class_ids.items():
            gt_percentages[class_name] = (
                100.0 * np.sum(ground_truth == class_id) / total_pixels
            )
            prediction_percentages[class_name] = (
                100.0 * np.sum(prediction == class_id) / total_pixels
            )

        correct_pixels = ground_truth == prediction
        incorrect_pixels = ~correct_pixels

        correct_pixel_percentage = (
            100.0 * correct_pixels.sum() / total_pixels
        )

        incorrect_pixel_percentage = (
            100.0 * incorrect_pixels.sum() / total_pixels
        )

        missed_object_pixels = (
            (ground_truth != background_id)
            & (prediction == background_id)
        )

        false_object_pixels = (
            (ground_truth == background_id)
            & (prediction != background_id)
        )

        wrong_object_class_pixels = (
            (ground_truth != background_id)
            & (prediction != background_id)
            & (ground_truth != prediction)
        )

        missed_object_percentage = (
            100.0 * missed_object_pixels.sum() / total_pixels
        )

        false_object_percentage = (
            100.0 * false_object_pixels.sum() / total_pixels
        )

        wrong_object_class_percentage = (
            100.0 * wrong_object_class_pixels.sum() / total_pixels
        )

        gt_object_classes = [
            class_name
            for class_name in object_class_names
            if np.any(ground_truth == class_ids[class_name])
        ]

        prediction_object_classes = [
            class_name
            for class_name in object_class_names
            if np.any(prediction == class_ids[class_name])
        ]

        missed_classes = [
            class_name
            for class_name in gt_object_classes
            if class_name not in prediction_object_classes
        ]

        false_classes = [
            class_name
            for class_name in prediction_object_classes
            if class_name not in gt_object_classes
        ]

        class_confusions = []

        for gt_class_name in object_class_names:
            gt_class_id = class_ids[gt_class_name]

            for prediction_class_name in object_class_names:
                prediction_class_id = class_ids[prediction_class_name]

                if gt_class_name == prediction_class_name:
                    continue

                confusion_pixels = np.logical_and(
                    ground_truth == gt_class_id,
                    prediction == prediction_class_id,
                ).sum()

                if confusion_pixels > 0:
                    class_confusions.append(
                        (
                            gt_class_name,
                            prediction_class_name,
                            int(confusion_pixels),
                        )
                    )

        class_confusions.sort(
            key=lambda item: item[2],
            reverse=True,
        )

        if missed_classes:
            error_type = "Непредсказание объекта"
        elif class_confusions:
            error_type = "Перепутан класс"
        elif false_classes:
            error_type = "Ложный объект"
        elif incorrect_pixel_percentage > 0:
            error_type = "Небольшая ошибка границ"
        else:
            error_type = "Корректное предсказание"

        if class_confusions:
            gt_confusion_class = class_confusions[0][0]
            prediction_confusion_class = class_confusions[0][1]
            confusion_text = (
                f"{gt_confusion_class} → "
                f"{prediction_confusion_class}"
            )
        else:
            confusion_text = ""

        results.append(
            {
                "image": image_name,
                "error_type": error_type,
                "confusion": confusion_text,
                "correct_pixels_percent": correct_pixel_percentage,
                "incorrect_pixels_percent": incorrect_pixel_percentage,
                "missed_object_percent": missed_object_percentage,
                "false_object_percent": false_object_percentage,
                "wrong_class_percent": wrong_object_class_percentage,
                "gt_background_percent": gt_percentages["background"],
                "gt_cat_percent": gt_percentages.get("cat", 0.0),
                "gt_dog_percent": gt_percentages.get("dog", 0.0),
                "prediction_background_percent": prediction_percentages[
                    "background"
                ],
                "prediction_cat_percent": prediction_percentages.get(
                    "cat",
                    0.0,
                ),
                "prediction_dog_percent": prediction_percentages.get(
                    "dog",
                    0.0,
                ),
                "missed_classes": ", ".join(missed_classes),
                "false_classes": ", ".join(false_classes),
            }
        )

    results_dataframe = pd.DataFrame(results)

    results_dataframe = results_dataframe.sort_values(
        [
            "incorrect_pixels_percent",
            "missed_object_percent",
            "wrong_class_percent",
        ],
        ascending=False,
    ).reset_index(drop=True)

    print("=" * 60)
    print("АНАЛИЗ ОШИБОК ПРЕДСКАЗАНИЙ")
    print("=" * 60)
    print(f"Проанализировано изображений: {len(results_dataframe)}")
    print(
        "Корректная классификация пикселей в среднем: "
        f"{results_dataframe['correct_pixels_percent'].mean():.2f}%"
    )
    print(
        "Ошибочная классификация пикселей в среднем: "
        f"{results_dataframe['incorrect_pixels_percent'].mean():.2f}%"
    )
    print(
        "Пропущенные объекты в среднем: "
        f"{results_dataframe['missed_object_percent'].mean():.2f}%"
    )
    print(
        "Ложные объекты в среднем: "
        f"{results_dataframe['false_object_percent'].mean():.2f}%"
    )
    print(
        "Перепутанные классы в среднем: "
        f"{results_dataframe['wrong_class_percent'].mean():.2f}%"
    )
    print("=" * 60)

    return results_dataframe


def show_prediction_comparisons(
    image_names,
    images_directory,
    ground_truth_directory,
    predictions_directory,
    predictions_analysis,
    class_ids,
    class_colors_bgr,
):
    """
    Показывает исходное изображение, Ground Truth и Prediction с долями классов.

    Args:
        image_names (list[str]):
            Список имён изображений без расширения.

        images_directory (Path):
            Директория с исходными изображениями.

        ground_truth_directory (Path):
            Директория с индексированными Ground Truth масками.

        predictions_directory (Path):
            Директория с предсказанными индексированными масками.

        predictions_analysis (pd.DataFrame):
            Таблица с индивидуальными Dice для изображений.

        class_ids (dict):
            Словарь с идентификаторами классов.

        class_colors_bgr (dict):
            Словарь цветов классов в формате BGR.

    Returns:
        None.
    """
    print("=" * 60)
    print("ВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ")
    print("=" * 60)
    print(f"Изображений для отображения: {len(image_names)}")

    for image_name in image_names:
        image_path = images_directory / f"{image_name}.jpg"
        ground_truth_path = ground_truth_directory / f"{image_name}.png"
        prediction_path = predictions_directory / f"{image_name}.png"

        image = cv2.imread(str(image_path))

        ground_truth = cv2.imread(
            str(ground_truth_path),
            cv2.IMREAD_GRAYSCALE,
        )

        prediction = cv2.imread(
            str(prediction_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise FileNotFoundError(
                f"Исходное изображение не найдено: {image_path}"
            )

        if ground_truth is None:
            raise FileNotFoundError(
                f"Ground Truth не найден: {ground_truth_path}"
            )

        if prediction is None:
            raise FileNotFoundError(
                f"Prediction не найден: {prediction_path}"
            )

        if image.shape[:2] != prediction.shape:
            raise ValueError(
                f"Размер изображения и Prediction не совпадает: {image_name}"
            )

        if ground_truth.shape != prediction.shape:
            raise ValueError(
                f"Размер Ground Truth и Prediction не совпадает: {image_name}"
            )

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        ground_truth_overlay = image_rgb.copy()
        prediction_overlay = image_rgb.copy()

        for class_name, class_id in class_ids.items():
            if class_name == "background":
                continue

            color_bgr = class_colors_bgr[class_name]

            color_rgb = np.array(
                [
                    color_bgr[2],
                    color_bgr[1],
                    color_bgr[0],
                ],
                dtype=np.uint8,
            )

            ground_truth_mask = ground_truth == class_id
            prediction_mask = prediction == class_id

            ground_truth_overlay[ground_truth_mask] = (
                0.55 * ground_truth_overlay[ground_truth_mask]
                + 0.45 * color_rgb
            ).astype(np.uint8)

            prediction_overlay[prediction_mask] = (
                0.55 * prediction_overlay[prediction_mask]
                + 0.45 * color_rgb
            ).astype(np.uint8)

        total_pixels = prediction.size

        ground_truth_class_percentages = {}
        prediction_class_percentages = {}

        for class_name, class_id in class_ids.items():
            ground_truth_pixels = int(
                (ground_truth == class_id).sum()
            )

            prediction_pixels = int(
                (prediction == class_id).sum()
            )

            ground_truth_class_percentages[class_name] = (
                100.0 * ground_truth_pixels / total_pixels
            )

            prediction_class_percentages[class_name] = (
                100.0 * prediction_pixels / total_pixels
            )

        image_metrics = predictions_analysis[
            predictions_analysis["image"] == image_name
        ]

        if len(image_metrics) != 1:
            raise ValueError(
                f"Для изображения {image_name} найдено "
                f"{len(image_metrics)} строк в predictions_analysis"
            )

        image_metrics = image_metrics.iloc[0]

        ground_truth_text = [
            (
                f"{class_name}: "
                f"{ground_truth_class_percentages[class_name]:.1f}%"
            )
            for class_name in class_ids
        ]

        prediction_text = [
            (
                f"{class_name}: "
                f"{prediction_class_percentages[class_name]:.1f}%"
            )
            for class_name in class_ids
        ]

        figure, axes = plt.subplots(
            1,
            3,
            figsize=(18, 6),
        )

        axes[0].imshow(image_rgb)
        axes[0].set_title(
            f"{image_name}\n"
            "Исходное изображение"
        )

        axes[1].imshow(ground_truth_overlay)
        axes[1].set_title(
            "Ground Truth\n"
            + "\n".join(ground_truth_text)
        )

        axes[2].imshow(prediction_overlay)
        axes[2].set_title(
            "Prediction\n"
            f"mDice = {image_metrics['object_mDice']:.4f}\n"
            f"cat Dice = {image_metrics['cat_dice']:.4f}\n"
            f"dog Dice = {image_metrics['dog_dice']:.4f}\n"
            + "\n".join(prediction_text)
        )

        for axis in axes:
            axis.axis("off")

        plt.tight_layout()
        plt.show()

        print(f"\n{image_name}:")
        print(
            f"mDice = {image_metrics['object_mDice']:.4f}, "
            f"cat Dice = {image_metrics['cat_dice']:.4f}, "
            f"dog Dice = {image_metrics['dog_dice']:.4f}"
        )

        print("Ground Truth:")
        for class_name in class_ids:
            print(
                f"Класс {class_name}: "
                f"{ground_truth_class_percentages[class_name]:.2f}% пикселей"
            )

        print("Prediction:")
        for class_name in class_ids:
            print(
                f"Класс {class_name}: "
                f"{prediction_class_percentages[class_name]:.2f}% пикселей"
            )

    print("=" * 60)