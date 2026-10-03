_base_ = ["./cats_dogs.py"]


# Pipeline для обучения без аугментаций
train_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="PackSegInputs"),
        ],
    ),
)