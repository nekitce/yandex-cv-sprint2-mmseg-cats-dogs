_base_ = [
    "../_base_/models/pspnet_unet_s5-d16.py",
    "../_base_/datasets/cats_dogs_no_aug.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/cats_dogs_schedule.py",
]

crop_size = (256, 256)

data_preprocessor = dict(
    size=crop_size,
)

model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(
        mode="whole",
    ),
    decode_head=dict(
        num_classes=3,
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=1.0,
            ),
            dict(
                type="DiceLoss",
                loss_weight=4.0,
            ),
        ],
    ),
    auxiliary_head=dict(
        num_classes=3,
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=1.0,
            ),
            dict(
                type="DiceLoss",
                loss_weight=4.0,
            ),
        ],
    ),
)