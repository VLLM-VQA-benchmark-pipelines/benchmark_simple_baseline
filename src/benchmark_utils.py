import os

import pandas as pd

from dataset_utils import DatasetIterator
from metric_evaluator import MetricEvaluator
from model_utils import Qwen2VL_model


def get_ouptut_csv_path(model_name, dataset_name, output_dir_path, output_type):
    if output_type not in ("answers", "metrics_by_id", "metrics_by_doc_type"):
        raise ValueError(
            'output_type должен быть равен "answers", "metrics_by_id", "metrics_by_doc_type"'
        )

    answers_csv_filename = f"{model_name}_{dataset_name}_{output_type}.csv"
    answear_csv_path = os.path.join(output_dir_path, answers_csv_filename)
    return answear_csv_path


def get_model_answers_on_VQA_dataset(
    model_name,
    dataset_name,
    datasets_dir_path,
    answers_dir_path,
    cache_directory,
    iter_log=True,
):
    print("\n******************************************")
    print("Начинаем бенчмарк")
    print(f"модели: {model_name}")
    print(f"на наборе данных: {dataset_name}\n")

    # создаем модель
    model = Qwen2VL_model(cache_directory, model_name)

    # создаем итератор
    dataset_dir_path = os.path.join(datasets_dir_path, dataset_name)
    iterator = DatasetIterator(dataset_dir_path)

    # создаем массив для хранения ответов от модели
    model_answers = []

    if iter_log:
        print("\nИтерируемся по набору данных")
    for image_path, question, answear in iterator:
        model_answer = model.predict(image=image_path, question=question)
        model_answers.append(model_answer)

        # отладочный вывод
        if iter_log:
            print(
                f"{iterator.row_index} | {image_path} | {question} | {answear} | {model_answer} |"
            )

    # Сохраняем csv-таблицу ответов от модели
    df = pd.DataFrame(model_answers, columns=["answer"])
    answear_csv_path = get_ouptut_csv_path(
        model_name, dataset_name, answers_dir_path, output_type="answers"
    )
    df.to_csv(answear_csv_path, sep=";", encoding="utf-8-sig", index=False)

    print(f"\nОтветы от модели сохранены в: {answear_csv_path}")


def evaluate_model_metrics_by_answers(
    model_name, dataset_name, datasets_dir_path, answers_dir_path, metrics_dir_path
):
    print("\nОцениваем метрики по ответам модели")
    dataset_csv_path = os.path.join(datasets_dir_path, dataset_name, "annotation.csv")
    answear_csv_path = get_ouptut_csv_path(
        model_name, dataset_name, answers_dir_path, "answers"
    )

    # Оцениваем метрики
    metric_eval = MetricEvaluator(dataset_csv_path, answear_csv_path)
    df_metrics_by_id = metric_eval.calculate_metrics_by_id()
    df_metrics_by_doc_type = metric_eval.calculate_metrics_by_doc_type(df_metrics_by_id)

    metrics_by_id_csv_path = get_ouptut_csv_path(
        model_name, dataset_name, metrics_dir_path, "metrics_by_id"
    )
    df_metrics_by_id.to_csv(
        metrics_by_id_csv_path, sep=";", encoding="utf-8-sig", index=False
    )
    print(f"Метрики по ID сохранены в: {metrics_by_id_csv_path}")

    metrics_by_doc_type_csv_path = get_ouptut_csv_path(
        model_name, dataset_name, metrics_dir_path, "metrics_by_doc_type"
    )
    df_metrics_by_doc_type.to_csv(
        metrics_by_doc_type_csv_path, sep=";", encoding="utf-8-sig", index=False
    )
    print(f"Метрики по типу документов сохранены в: {metrics_by_doc_type_csv_path}")
