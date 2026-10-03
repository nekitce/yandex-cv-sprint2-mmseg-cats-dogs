"""
Функции для анализа метрик экспериментов MMSegmentation.
"""

# Стандартная библиотека
from pathlib import Path

# Сторонние библиотеки
from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd


def find_latest_experiment_directory(results_root):
    """
    Находит последний эксперимент по дате и времени в имени директории.

    Args:
        results_root (Path):
            Корневая директория результатов экспериментов.

    Returns:
        Path:
            Директория последнего эксперимента.
    """
    experiment_directories = []

    for path in results_root.iterdir():
        if not path.is_dir():
            continue

        try:
            experiment_datetime = pd.to_datetime(
                path.name,
                format="%Y%m%d_%H%M%S",
            )
        except ValueError:
            continue

        experiment_directories.append(
            (experiment_datetime, path)
        )

    if not experiment_directories:
        raise FileNotFoundError(
            f"В {results_root} не найдено ни одной "
            "директории эксперимента формата YYYYMMDD_HHMMSS."
        )

    experiment_directories.sort(
        key=lambda item: item[0]
    )

    latest_experiment_datetime, latest_experiment = (
        experiment_directories[-1]
    )

    print("=" * 60)
    print("ПОСЛЕДНИЙ ЭКСПЕРИМЕНТ")
    print("=" * 60)
    print(f"Корневая директория: {results_root}")
    print(
        f"Найдено экспериментов: "
        f"{len(experiment_directories)}"
    )
    print(f"Последний эксперимент: {latest_experiment}")
    print(
        f"Дата и время: "
        f"{latest_experiment_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("=" * 60)

    return latest_experiment


def find_training_json_log(results_root):
    """
    Находит единственный JSONL-лог последнего валидного эксперимента.

    Args:
        results_root (Path):
            Корневая директория результатов экспериментов.

    Returns:
        Path:
            Путь к JSONL-логу обучения.
    """
    latest_experiment = find_latest_experiment_directory(
        results_root
    )

    json_logs = sorted(
        latest_experiment.glob("vis_data/*.json")
    )

    if len(json_logs) != 1:
        raise RuntimeError(
            "Ожидался ровно один JSONL-лог в vis_data, "
            f"но найдено: {len(json_logs)}\n"
            f"Эксперимент: {latest_experiment}"
        )

    log_path = json_logs[0]

    print("=" * 60)
    print("JSONL-ЛОГ ОБУЧЕНИЯ")
    print("=" * 60)
    print(f"Эксперимент: {latest_experiment}")
    print(f"JSONL: {log_path}")
    print(f"Файл существует: {log_path.exists()}")
    print("=" * 60)

    return log_path


def plot_mmengine_training_metrics(
    log_path,
    iterations_per_epoch,
):
    """
    Строит полный набор графиков всех числовых метрик train и validation.

    Args:
        log_path (Path):
            Путь к JSONL-логу MMEngine.
        iterations_per_epoch (int):
            Количество итераций обучения в одной эпохе.

    Returns:
        None.
    """
    if not log_path.exists():
        raise FileNotFoundError(
            f"JSONL-лог не найден: {log_path}"
        )

    log_records = pd.read_json(
        log_path,
        lines=True,
    )

    train_records = log_records[
        log_records["loss"].notna()
    ].copy()

    val_records = log_records[
        log_records["mDice"].notna()
    ].copy()

    if train_records.empty:
        raise RuntimeError(
            "В JSONL-логе не найдены train-записи с метрикой loss."
        )

    if val_records.empty:
        raise RuntimeError(
            "В JSONL-логе не найдены validation-записи с метрикой mDice."
        )

    train_records["epoch"] = (
        train_records["step"] / iterations_per_epoch
    )

    val_records["epoch"] = range(
        1,
        len(val_records) + 1,
    )

    excluded_columns = {
        "epoch",
        "step",
        "iter",
    }

    train_metric_columns = [
        column
        for column in train_records.columns
        if (
            column not in excluded_columns
            and pd.api.types.is_numeric_dtype(
                train_records[column]
            )
            and train_records[column].notna().any()
        )
    ]

    val_metric_columns = [
        column
        for column in val_records.columns
        if (
            column not in excluded_columns
            and pd.api.types.is_numeric_dtype(
                val_records[column]
            )
            and val_records[column].notna().any()
        )
    ]

    print("=" * 60)
    print("АНАЛИЗ МЕТРИК ОБУЧЕНИЯ")
    print("=" * 60)
    print(f"JSONL: {log_path}")
    print(f"Всего записей: {len(log_records)}")
    print(f"Train-записей: {len(train_records)}")
    print(f"Validation-записей: {len(val_records)}")
    print(
        f"Train-метрик для графиков: "
        f"{len(train_metric_columns)}"
    )
    print(
        f"Validation-метрик для графиков: "
        f"{len(val_metric_columns)}"
    )
    print("=" * 60)

    print("TRAIN METRICS:")
    for metric_name in train_metric_columns:
        metric_values = train_records[metric_name].dropna()

        first_value = metric_values.iloc[0]
        last_value = metric_values.iloc[-1]
        minimum_value = metric_values.min()
        maximum_value = metric_values.max()

        print(
            f"{metric_name}: "
            f"первое={first_value:.6f}, "
            f"последнее={last_value:.6f}, "
            f"минимум={minimum_value:.6f}, "
            f"максимум={maximum_value:.6f}"
        )

    print("=" * 60)
    print("VALIDATION METRICS:")

    for metric_name in val_metric_columns:
        metric_values = val_records[metric_name].dropna()

        first_value = metric_values.iloc[0]
        last_value = metric_values.iloc[-1]

        if metric_name in [
            "mDice",
            "mIoU",
            "mAcc",
            "aAcc",
        ]:
            best_index = metric_values.idxmax()
            best_value = metric_values.loc[best_index]
            best_epoch = val_records.loc[
                best_index,
                "epoch",
            ]

            print(
                f"{metric_name}: "
                f"первое={first_value:.2f}, "
                f"последнее={last_value:.2f}, "
                f"лучшее={best_value:.2f} "
                f"(epoch {best_epoch:.0f})"
            )
        else:
            minimum_value = metric_values.min()
            maximum_value = metric_values.max()

            print(
                f"{metric_name}: "
                f"первое={first_value:.6f}, "
                f"последнее={last_value:.6f}, "
                f"минимум={minimum_value:.6f}, "
                f"максимум={maximum_value:.6f}"
            )

    print("=" * 60)

    if "mDice" in val_records.columns:
        best_mdice_index = val_records["mDice"].idxmax()
        best_mdice = val_records.loc[
            best_mdice_index,
            "mDice",
        ]
        best_mdice_epoch = val_records.loc[
            best_mdice_index,
            "epoch",
        ]

        print("ЛУЧШИЙ VALIDATION MDICE")
        print(
            f"mDice={best_mdice:.2f} "
            f"на epoch {best_mdice_epoch:.0f}"
        )
        print("=" * 60)

    all_train_metrics = train_metric_columns
    all_val_metrics = val_metric_columns

    train_figures_count = len(all_train_metrics)
    val_figures_count = len(all_val_metrics)

    print(
        f"Будет построено train-графиков: "
        f"{train_figures_count}"
    )
    print(
        f"Будет построено validation-графиков: "
        f"{val_figures_count}"
    )
    print("=" * 60)

    if all_train_metrics:
        train_rows = (
            train_figures_count + 1
        ) // 2

        fig, axes = plt.subplots(
            train_rows,
            2,
            figsize=(16, 4 * train_rows),
            squeeze=False,
        )

        for index, metric_name in enumerate(
            all_train_metrics
        ):
            row = index // 2
            column = index % 2

            metric_values = train_records[
                metric_name
            ]

            axes[row, column].plot(
                train_records["epoch"],
                metric_values,
                label=metric_name,
            )

            axes[row, column].set_title(
                f"Train: {metric_name}"
            )
            axes[row, column].set_xlabel(
                "Epoch"
            )
            axes[row, column].set_ylabel(
                metric_name
            )
            axes[row, column].grid()
            axes[row, column].legend()

        if train_figures_count % 2 != 0:
            axes[-1, -1].axis("off")

        fig.suptitle(
            "Все train-метрики",
            fontsize=16,
        )

        plt.tight_layout()
        plt.show()

    if all_val_metrics:
        val_rows = (
            val_figures_count + 1
        ) // 2

        fig, axes = plt.subplots(
            val_rows,
            2,
            figsize=(16, 4 * val_rows),
            squeeze=False,
        )

        for index, metric_name in enumerate(
            all_val_metrics
        ):
            row = index // 2
            column = index % 2

            metric_values = val_records[
                metric_name
            ]

            axes[row, column].plot(
                val_records["epoch"],
                metric_values,
                label=metric_name,
            )

            axes[row, column].set_title(
                f"Validation: {metric_name}"
            )
            axes[row, column].set_xlabel(
                "Epoch"
            )
            axes[row, column].set_ylabel(
                metric_name
            )
            axes[row, column].grid()
            axes[row, column].legend()

        if val_figures_count % 2 != 0:
            axes[-1, -1].axis("off")

        fig.suptitle(
            "Все validation-метрики",
            fontsize=16,
        )

        plt.tight_layout()
        plt.show()

    print("=" * 60)
    print("АНАЛИЗ МЕТРИК ЗАВЕРШЁН")
    print("=" * 60)


def compare_segmentation_experiments(
    experiment_names,
    log_paths,
    test_results,
):
    """
    Формирует итоговую таблицу сравнения произвольного количества экспериментов.

    Args:
        experiment_names (list[str]):
            Названия сравниваемых экспериментов.

        log_paths (list[Path]):
            Пути к JSONL-логам соответствующих экспериментов.

        test_results (list[dict | list]):
            Результаты тестирования соответствующих экспериментов.

    Returns:
        pd.DataFrame:
            Итоговая таблица сравнения train, validation и test метрик.
    """
    if not (
        len(experiment_names)
        == len(log_paths)
        == len(test_results)
    ):
        raise ValueError(
            "Количество названий экспериментов, логов и результатов "
            "тестирования должно совпадать."
        )

    comparison_rows = []

    for experiment_name, log_path, experiment_test_results in zip(
        experiment_names,
        log_paths,
        test_results,
    ):
        if not log_path.exists():
            raise FileNotFoundError(
                f"JSONL-лог не найден: {log_path}"
            )

        log_records = pd.read_json(
            log_path,
            lines=True,
        )

        train_records = log_records[
            log_records["loss"].notna()
        ].copy()

        val_records = log_records[
            log_records["mDice"].notna()
        ].copy()

        if train_records.empty:
            raise RuntimeError(
                f"В логе {log_path} не найдены train-записи."
            )

        if val_records.empty:
            raise RuntimeError(
                f"В логе {log_path} не найдены validation-записи."
            )

        best_val_index = val_records["mDice"].idxmax()
        best_val_record = val_records.loc[best_val_index]

        test_result = experiment_test_results

        if isinstance(test_result, list):
            if len(test_result) != 1:
                raise RuntimeError(
                    f"Для {experiment_name} ожидался один словарь "
                    f"test-результатов, получено: {len(test_result)}."
                )

            test_result = test_result[0]

        if not isinstance(test_result, dict):
            raise TypeError(
                f"Test-результат {experiment_name} должен быть словарём."
            )

        comparison_rows.append(
            {
                "Модель": experiment_name,
                "Train loss": train_records["loss"].iloc[-1],
                "Train loss min": train_records["loss"].min(),
                "Train loss max": train_records["loss"].max(),
                "Val mDice": best_val_record.get("mDice"),
                "Val mIoU": best_val_record.get("mIoU"),
                "Val mAcc": best_val_record.get("mAcc"),
                "Val aAcc": best_val_record.get("aAcc"),
                "Val epoch": best_val_record.get("epoch"),
                "Test mDice": test_result.get("mDice"),
                "Test mIoU": test_result.get("mIoU"),
                "Test mAcc": test_result.get("mAcc"),
                "Test aAcc": test_result.get("aAcc"),
            }
        )

    comparison_table = pd.DataFrame(comparison_rows)

    numeric_columns = comparison_table.select_dtypes(
        include="number"
    ).columns

    comparison_table[numeric_columns] = comparison_table[
        numeric_columns
    ].round(4)

    print("=" * 60)
    print("ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)
    print(f"Сравниваемых моделей: {len(comparison_table)}")
    print("=" * 60)

    display(comparison_table)

    print("=" * 60)
    print("СРАВНЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)

    return comparison_table