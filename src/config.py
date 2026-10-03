# Стандартная библиотека Python
from pathlib import Path


class Config:
    """
    Хранит конфигурацию проекта семантической сегментации.

    Args:
        None

    Returns:
        None.
    """

    # Репозиторий проекта
    GITHUB_REPOSITORY = ""

    # Корневая директория проекта mmsegmentation
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Директория данных
    DATA_ROOT = PROJECT_ROOT / "data"

    # Директория исходного датасета
    RAW_DATASET_ROOT = DATA_ROOT / "raw_dataset"

    # Директория исходных изображений
    RAW_IMAGES_ROOT = RAW_DATASET_ROOT / "img"

    # Директория исходных масок
    RAW_LABELS_ROOT = RAW_DATASET_ROOT / "labels"

    # Директории изображений по выборкам
    TRAIN_IMAGES_ROOT = RAW_IMAGES_ROOT / "train"
    VAL_IMAGES_ROOT = RAW_IMAGES_ROOT / "val"
    TEST_IMAGES_ROOT = RAW_IMAGES_ROOT / "test"

    # Директории исходных масок по выборкам
    TRAIN_LABELS_ROOT = RAW_LABELS_ROOT / "train"
    VAL_LABELS_ROOT = RAW_LABELS_ROOT / "val"
    TEST_LABELS_ROOT = RAW_LABELS_ROOT / "test"

    # Директория исправленных цветных масок из CVAT
    CORRECTED_LABELS_ROOT = DATA_ROOT / "correct_mask"

    # Директории исправленных цветных масок по выборкам
    TRAIN_CORRECTED_LABELS_ROOT = CORRECTED_LABELS_ROOT / "train"
    VAL_CORRECTED_LABELS_ROOT = CORRECTED_LABELS_ROOT / "val"
    TEST_CORRECTED_LABELS_ROOT = CORRECTED_LABELS_ROOT / "test"

    # Директория индексных масок для MMSeg
    INDEXED_LABELS_ROOT = DATA_ROOT / "correct_mask_indexed"

    # Директории индексных масок по выборкам
    TRAIN_INDEXED_LABELS_ROOT = INDEXED_LABELS_ROOT / "train"
    VAL_INDEXED_LABELS_ROOT = INDEXED_LABELS_ROOT / "val"
    TEST_INDEXED_LABELS_ROOT = INDEXED_LABELS_ROOT / "test"

    # Названия классов семантической сегментации
    CLASS_NAMES = ("background", "cat", "dog")

    # Числовые индексы классов
    CLASS_IDS = {
        "background": 0,
        "cat": 1,
        "dog": 2,
    }

    # Количество классов
    NUM_CLASSES = len(CLASS_NAMES)

    # Размер входного изображения модели
    INPUT_SIZE = (256, 256)

    # Цвета масок CVAT в формате BGR для cv2.imread
    CVAT_MASK_COLORS_BGR = {
        "background": (0, 0, 0),
        "cat": (18, 122, 225),
        "dog": (240, 120, 140),
    }

    # Директория изображений с наложенными масками
    MASK_OVERLAYS_ROOT = DATA_ROOT / "mask_overlays"

    # Директории визуализаций масок по выборкам
    TRAIN_MASK_OVERLAYS_ROOT = MASK_OVERLAYS_ROOT / "train"
    VAL_MASK_OVERLAYS_ROOT = MASK_OVERLAYS_ROOT / "val"
    TEST_MASK_OVERLAYS_ROOT = MASK_OVERLAYS_ROOT / "test"

    # Директория визуализаций плохих масок
    BAD_MASKS_ROOT = DATA_ROOT / "bad_masks"

    # Визуализации плохих масок по выборкам
    TRAIN_BAD_MASKS_ROOT = BAD_MASKS_ROOT / "train"
    VAL_BAD_MASKS_ROOT = BAD_MASKS_ROOT / "val"
    TEST_BAD_MASKS_ROOT = BAD_MASKS_ROOT / "test"

    # Списки изображений с проблемной разметкой
    BAD_MASKS_TRAIN = [
        "000000023731_404.jpg",
        "000000028253_7169.jpg",
        "000000049758_3963.jpg",
        "000000066011_2187.jpg",
        "000000118680_4349.jpg",
        "000000121530_5761.jpg",
        "000000216665_5491.jpg",
        "000000247301_4455.jpg",
        "000000258305_3996.jpg",
        "000000275028_3168.jpg",
        "000000275919_4499.jpg",
        "000000317781_4461.jpg",
        "000000325768_1316.jpg",
        "000000326073_4335.jpg",
        "000000419618_7033.jpg",
        "000000481212_908.jpg",
        "000000562835_2386.jpg",
        "000000574769_0.jpg",
    ]

    BAD_MASKS_VAL = []
    BAD_MASKS_TEST = []

    # Директория checkpoint
    CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"

    # Директория артефактов
    ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

    # Директория архивов аннотаций для CVAT
    CVAT_ANNOTATIONS_ROOT = ARTIFACTS_ROOT / "cvat_annotations"

    # ZIP-архивы аннотаций для CVAT
    TRAIN_CVAT_ANNOTATIONS_PATH = (
        CVAT_ANNOTATIONS_ROOT / "train_annotations.zip"
    )
    VAL_CVAT_ANNOTATIONS_PATH = (
        CVAT_ANNOTATIONS_ROOT / "val_annotations.zip"
    )
    TEST_CVAT_ANNOTATIONS_PATH = (
        CVAT_ANNOTATIONS_ROOT / "test_annotations.zip"
    )

    # Директория результатов метрик
    METRICS_ROOT = ARTIFACTS_ROOT / "metrics"

    # Итоговый отчёт
    REPORT_PATH = PROJECT_ROOT / "README.md"

    # Конфигурации MMSegmentation
    MMSeg_CONFIGS_ROOT = PROJECT_ROOT / "configs"

    # Конфигурации экспериментов cats/dogs
    CATS_DOGS_CONFIGS_ROOT = MMSeg_CONFIGS_ROOT / "cats_dogs"

    # Конфигурации стартовых гипотез Stage 2
    STAGE_2_CONFIGS_ROOT = CATS_DOGS_CONFIGS_ROOT

    # Результаты экспериментов Stage 2
    STAGE_2_RESULTS_ROOT = ARTIFACTS_ROOT / "stage_2"

    # Результаты экспериментов Stage 3
    STAGE_3_RESULTS_ROOT = ARTIFACTS_ROOT / "stage_3"

    # Названия стартовых гипотез
    HYPOTHESIS_1_NAME = "unet_fcn_aug"
    HYPOTHESIS_2_NAME = "unet_pspnet_aug"
    HYPOTHESIS_3_NAME = "unet_pspnet_aug_exp2"

    # Название эксперимента HRNet-W18 с pretrained и fine-tuning
    HRNET_W18_FINETUNE_NAME = "hrnet_w18_finetune"

    # Контрольный baseline UNet + FCN без аугментаций
    BASELINE_NO_AUG_NAME = "unet_fcn_no_aug"

    # Конфигурация контрольного baseline UNet + FCN без аугментаций
    BASELINE_NO_AUG_CONFIG_PATH = (
        STAGE_2_CONFIGS_ROOT / f"{BASELINE_NO_AUG_NAME}.py"
    )

    # Контрольный baseline UNet + PSPNet без аугментаций
    BASELINE_PSPNET_NO_AUG_NAME = "unet_pspnet_no_aug"

    # Конфигурация контрольного baseline UNet + PSPNet без аугментаций
    BASELINE_PSPNET_NO_AUG_CONFIG_PATH = (
        STAGE_2_CONFIGS_ROOT / f"{BASELINE_PSPNET_NO_AUG_NAME}.py"
    )

    # Конфигурации стартовых гипотез
    HYPOTHESIS_1_CONFIG_PATH = (
        STAGE_2_CONFIGS_ROOT / f"{HYPOTHESIS_1_NAME}.py"
    )

    HYPOTHESIS_2_CONFIG_PATH = (
        STAGE_2_CONFIGS_ROOT / f"{HYPOTHESIS_2_NAME}.py"
    )

    # Конфигурация второго эксперимента Stage 3
    HYPOTHESIS_3_CONFIG_PATH = (
        CATS_DOGS_CONFIGS_ROOT / f"{HYPOTHESIS_3_NAME}.py"
    )

    # Конфигурация эксперимента HRNet-W18 с pretrained и fine-tuning
    HRNET_W18_FINETUNE_CONFIG_PATH = (
        CATS_DOGS_CONFIGS_ROOT / "hrnet_fcn_finetune.py"
    )

    # Параметры обучения
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    MAX_EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.01

    # Прозрачность маски при визуализации
    MASK_OVERLAY_ALPHA = 0.50

    # Seed для воспроизводимости
    SEED = 42