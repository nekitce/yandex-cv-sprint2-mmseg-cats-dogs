_base_ = ["./cats_dogs.py"]


# Pipeline для обучения с аугментациями Stage 2
train_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="PhotoMetricDistortion"),
            dict(type="RandomRotate", prob=0.5, degree=(-45, 45)),
            dict(
                type="RandomCutOut",
                prob=0.4,
                n_holes=(7, 15),
                cutout_ratio=(0.1, 0.15),
            ),
            dict(
                type="Albu",
                transforms=[
                    dict(
                        type="GridDistortion",
                        num_steps=10,
                        p=1,
                    ),
                ],
            ),
            dict(type="PackSegInputs"),
        ],
    ),
)