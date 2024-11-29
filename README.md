# Описание

Пишем первый простой бенчмарк для модели на локальных данных.

# Данные для бенчмарка

Заготовка для бенчмарка запускается на датасетах:
```
data
├── MIDV-2020_MINI
│   ├── annotation.csv
│   └── images
│
├── MTVQA_TEST_RU_MINI_0_9
│   ├── annotation.tsv
│   └── images
│
└── MTVQA_TEST_RU_MINI_10_19
    ├── annotation.tsv
    └── images
```

Скачать их можно по ссылкам приведенным в описании ([ссылка](https://github.com/VLLM-VQA-benchmark-pipelines/wiki/blob/main/projects/tasks/5.%20Готово/Подготовка%20тестовых%20мини%20датасетов.md)).

# Docker контейнер

## Build Docker image

Для сборки `Docker image` выполним команду:
```
docker build -t simple_bench:0.0.1 -f docker/Dockerfile-cu124 .
```

## Run Docker Container

Для запуска `Docker Container` выполним команду:
```
docker run \
  --gpus all \
  -it \
  -v ./src:/workspace \
  -v ./data:/workspace/data \
  simple_bench:0.0.1
```

Нам откроется терминал внутри `Docker Container`.

Для запуска бенчмарка выполним в нем команду:
```
python run_benchmark.py
```

### Пример вывода консоли
```
{
    "datasets": {
        "MIDV-2020_MINI": "data/datasets/MIDV-2020_MINI"
    },
    "models": {
        "Qwen2-VL-2B-Instruct": "",
        "Qwen2-VL-2B-Instruct-GPTQ-Int8": ""
    },
    "metrics": [
        "WER",
        "CER",
        "BLEU"
    ]
}
Configuration loaded from file: config.json
Валидация конфигурационного файла: config.json
Конфигурационный файл корректен

******************************************
Начинаем бенчмарк
модели: Qwen2-VL-2B-Instruct
на наборе данных: MIDV-2020_MINI

Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  2.63it/s]

Итерируемся по набору данных
1 | data/datasets/MIDV-2020_MINI/images/00.jpg | Напиши только фамилию! | Карибжанов | Карибжанов |
2 | data/datasets/MIDV-2020_MINI/images/00.jpg | Напиши только имя! | Леонид | Карипжанов |
3 | data/datasets/MIDV-2020_MINI/images/00.jpg | Напиши только отчество | Давидович | Отчество: ЛЕОНИД ДАВИДОВИЧ |
4 | data/datasets/MIDV-2020_MINI/images/00.jpg | Напиши только дату рождения ! | 03.06.1987 | 03.06.1978 |
5 | data/datasets/MIDV-2020_MINI/images/00.jpg | Напиши только номер удостоверения! | 4 598 647 461 | 45 98 647461 |
6 | data/datasets/MIDV-2020_MINI/images/01.jpg | Напиши только фамилию! | Александрова | Александрова |
7 | data/datasets/MIDV-2020_MINI/images/01.jpg | Напиши только имя! | Александра | Александрова |
8 | data/datasets/MIDV-2020_MINI/images/01.jpg | Напиши только отчество | Александровна | Отчество - АЛЕКСАНДРОВНА |
9 | data/datasets/MIDV-2020_MINI/images/01.jpg | Напиши только дату рождения ! | 02.03.1987 | Дата рождения: 02.03.1987 |
10 | data/datasets/MIDV-2020_MINI/images/01.jpg | Напиши только номер удостоверения! | 241456789 | 02 41 456789 |

Ответы от модели сохранены в: data/models_answers/Qwen2-VL-2B-Instruct_MIDV-2020_MINI_answers.csv

Оцениваем метрики по ответам модели
Метрики по ID сохранены в: data/models_metrics/Qwen2-VL-2B-Instruct_MIDV-2020_MINI_metrics_by_id.csv
Метрики по типу документов сохранены в: data/models_metrics/Qwen2-VL-2B-Instruct_MIDV-2020_MINI_metrics_by_doc_type.csv

******************************************
Начинаем бенчмарк
модели: Qwen2-VL-2B-Instruct-GPTQ-Int8
на наборе данных: MIDV-2020_MINI

Итерируемся по набору данных
1 | data/datasets/MIDV-2020_MINI/images/00.jpg | Напиши только фамилию! | Карибжанов | Карибжанов |
2 | data/datasets/MIDV-2020_MINI/images/00.jpg | Напиши только имя! | Леонид | Карипжанов |
3 | data/datasets/MIDV-2020_MINI/images/00.jpg | Напиши только отчество | Давидович | Давидович |
4 | data/datasets/MIDV-2020_MINI/images/00.jpg | Напиши только дату рождения ! | 03.06.1987 | 03.06.1978 |
5 | data/datasets/MIDV-2020_MINI/images/00.jpg | Напиши только номер удостоверения! | 4 598 647 461 | 45 98 647461 |
6 | data/datasets/MIDV-2020_MINI/images/01.jpg | Напиши только фамилию! | Александрова | Александрова |
7 | data/datasets/MIDV-2020_MINI/images/01.jpg | Напиши только имя! | Александра | Александрова |
8 | data/datasets/MIDV-2020_MINI/images/01.jpg | Напиши только отчество | Александровна | Отчество Александровна |
9 | data/datasets/MIDV-2020_MINI/images/01.jpg | Напиши только дату рождения ! | 02.03.1987 | Дата рождения: 02.03.1987 |
10 | data/datasets/MIDV-2020_MINI/images/01.jpg | Напиши только номер удостоверения! | 241456789 | 02 41 456789 |

Ответы от модели сохранены в: data/models_answers/Qwen2-VL-2B-Instruct-GPTQ-Int8_MIDV-2020_MINI_answers.csv

Оцениваем метрики по ответам модели
Метрики по ID сохранены в: data/models_metrics/Qwen2-VL-2B-Instruct-GPTQ-Int8_MIDV-2020_MINI_metrics_by_id.csv
Метрики по типу документов сохранены в: data/models_metrics/Qwen2-VL-2B-Instruct-GPTQ-Int8_MIDV-2020_MINI_metrics_by_doc_type.csv
```

