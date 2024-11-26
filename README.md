# Описание

Пишем первый простой бенчмарк для модели на локальных данных.

# Docker контейнер

## Build Docker image

Для сборки `Docker image` выполним команду:
```
docker build -t qwenvl:2-cu124 -f docker/Dockerfile-cu124 .
```

## Run Docker Container

Для запуска `Docker Container` выполним команду:
```
 docker run \
    --gpus all \
    --rm \
    -it \
    -v ./src:/workspace \
    qwenvl:2-cu124
```

Нам откроется терминал внутри `Docker Container`.

Для запуска бенчмарка выполним в нем команду:
```
cd cd workspace
python run_benchmark.py
```

