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

