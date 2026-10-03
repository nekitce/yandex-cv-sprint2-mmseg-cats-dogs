# Тип датасета проекта
dataset_type = "CatsDogsDataset"

# Корневая директория датасета
data_root = "data"

# Размер входного изображения
crop_size = (256, 256)

# Количество картинок в batch
batch_size = 4


# Базовый pipeline для обучения
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]


# Pipeline для валидации и тестирования
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]


# Датасет для обучения
train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path="raw_dataset/img/train",
        seg_map_path="correct_mask_indexed/train",
    ),
    pipeline=train_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png",
)


# DataLoader для обучения
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(
        type="DefaultSampler",
        shuffle=True,
    ),
    dataset=train_dataset,
)


# Датасет для валидации
val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path="raw_dataset/img/val",
        seg_map_path="correct_mask_indexed/val",
    ),
    pipeline=test_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png",
)


# DataLoader для валидации
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),
    dataset=val_dataset,
)


# Датасет для тестирования
test_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path="raw_dataset/img/test",
        seg_map_path="correct_mask_indexed/test",
    ),
    pipeline=test_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png",
)


# DataLoader для тестирования
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),
    dataset=test_dataset,
)


# Метрики валидации
val_evaluator = dict(
    type="IoUMetric",
    iou_metrics=["mDice", "mIoU"],
)


# Метрики тестирования
test_evaluator = dict(
    type="IoUMetric",
    iou_metrics=["mDice", "mIoU"],
)