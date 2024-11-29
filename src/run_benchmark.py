import os

from scheduler_utils import BenchmarkScheduler


if __name__ == "__main__":
    cache_directory = "model_cache"
    datasets_dir_path = "data/datasets"
    answers_dir_path = "data/models_answers"
    metrics_dir_path = "data/models_metrics"
    config_path = "config.json"

    # создаем дирректории
    os.makedirs(answers_dir_path, exist_ok=True)
    os.makedirs(metrics_dir_path, exist_ok=True)

    # Сохраняем модели Qwen2-VL в примонтированую папку
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_directory = os.path.join(script_dir, cache_directory)

    # Запуск бенчмарка
    benchmark_scheduler = BenchmarkScheduler(config_path, display_json=True)
    benchmark_scheduler.run_scheduler(
        datasets_dir_path, answers_dir_path, cache_directory, metrics_dir_path
    )
