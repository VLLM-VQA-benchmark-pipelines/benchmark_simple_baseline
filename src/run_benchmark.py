import os

from benchmark_utils import get_model_answers_on_VQA_dataset
from benchmark_utils import evaluate_model_metrics_by_answers


if __name__ == "__main__":
    # Укажите путь к директории, где хотите хранить модели
    cache_directory = "model_cache"
    datasets_dir_path = "data/datasets"
    answers_dir_path = "data/models_answers"
    metrics_dir_path = "data/models_metrics"

    # создаем дирректории
    os.makedirs(answers_dir_path, exist_ok=True)
    os.makedirs(metrics_dir_path, exist_ok=True)

    # Сохраняем модели Qwen2-VL в примонтированую папку
    # иначе замучаемся качать их на каждый запуск контейнера
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_directory = os.path.join(script_dir, cache_directory)

    # Блок кода для одного бенчмарка 1 модели на 1 датасете
    model_name = "Qwen2-VL-2B-Instruct-AWQ" # "Qwen2-VL-2B-Instruct"
    dataset_name = "MIDV-2020_MINI"

    # Получаем ответы "1 модели  на 1 датасете"
    get_model_answers_on_VQA_dataset(
        model_name,
        dataset_name,
        datasets_dir_path,
        answers_dir_path,
        cache_directory,
        iter_log=True,
    )

    # Оцениваем метрики "1 модели  на 1 датасете"
    evaluate_model_metrics_by_answers(
        model_name, dataset_name, datasets_dir_path, answers_dir_path, metrics_dir_path
    )
