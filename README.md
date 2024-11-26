# Описание

Пишем первый простой бенчмарк для модели на локальных данных.

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

