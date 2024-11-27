import os

from dataset_utils import DatasetIterator
from model_utils import Qwen2VL_model
from metrics_utils import calculate_metrics


if __name__ == "__main__":
    # Укажите путь к директории, где хотите хранить модели
    cache_directory = "model_cache"
    dataset_dir_path = "data/MIDV-2020_MINI"

    # Сохраняем модели Qwen2-VL в примонтированую папку
    # иначе замучаемся качать их на каждый запуск контейнера
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_directory = os.path.join(script_dir, cache_directory)

    # Здесь пишем Интерпретатор конфига
    # Получаем список датасетов (их всего 2)
    # Список моделей (только Qwen2-VL модели)
    # Организуем циклы

    # Блок кода для одного бенчмарка 1 модели на 1 датасете

    # создаем модель
    model = Qwen2VL_model(cache_directory)
    # создаем итератор
    iterator = DatasetIterator(dataset_dir_path)
    # пока берем 1 картинку и 1 вопрос - далее можем брать батч или пройтись форчиком 
    image_path, question, answear = next(iterator)

    # отдаем модели 1 картинку и 1 вопрос, получаем ответ
    model_answer = model.predict(image=image_path, question=question)

    print(answear, model_answer)
    
    # Здесь реализуем "Оценщик метрик"
    metrics = calculate_metrics([answear], [model_answer])
    print(metrics)
