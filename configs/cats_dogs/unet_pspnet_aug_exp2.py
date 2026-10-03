_base_ = [
    "./unet_pspnet_aug.py",
]


# Конфигурация цикла обучения.
train_cfg = dict(
    type="EpochBasedTrainLoop",
    val_interval=1,
)


# Конфигурация цикла валидации.
val_cfg = dict(
    type="ValLoop",
)


# Конфигурация цикла тестирования.
test_cfg = dict(
    type="TestLoop",
)


# Оптимизатор и его параметры.
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=0.0005,
        weight_decay=0.1,
    ),
    clip_grad=None,
)


# План изменения learning rate.
param_scheduler = [
    dict(
        type="PolyLR",
        eta_min=1e-5,
        power=0.9,
        begin=0,
        by_epoch=True,
    ),
]


# Основные hooks обучения.
default_hooks = dict(
    timer=dict(
        type="IterTimerHook",
    ),
    logger=dict(
        type="LoggerHook",
        interval=25,
    ),
    param_scheduler=dict(
        type="ParamSchedulerHook",
    ),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=10,
        max_keep_ckpts=3,
        save_best="mDice",
    ),
    sampler_seed=dict(
        type="DistSamplerSeedHook",
    ),
    visualization=dict(
        type="SegVisualizationHook",
        draw=True,
        interval=10,
    ),
)