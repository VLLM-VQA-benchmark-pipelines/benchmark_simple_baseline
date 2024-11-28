import os

import pandas as pd


class DatasetIterator:
    """Provides methods for iterating over VQA datasets"""

    def __init__(self, dataset_dir_path, start=0):
        self.row_index = start
        self.dataset_dir_path = dataset_dir_path
        self._read_data()

    def _read_data(self):
        """Загружаем таблицу с аннотацией данных"""
        annot_path = os.path.join(self.dataset_dir_path, "annotation.csv")
        # считываем названия столбцов
        dataframe_header = pd.read_csv(annot_path, sep=";", nrows=1)

        # считываем содержание таблицы, начиная с self.row_index
        if self.row_index > 0:
            self.row_index -= 1

        dataframe = pd.read_csv(annot_path, sep=";", skiprows=self.row_index)

        # Записываем названия столбцов в dataframe из dataframe_header
        dataframe.columns = dataframe_header.columns
        self.iterator = dataframe.iterrows()

    def __iter__(self):
        return self

    def __next__(self):
        """Возвращаем склееный путь до изображения, вопрос и ответ на него"""
        # итерируемся по Dataframe
        index, row = next(self.iterator)
        self.row_index += 1

        # получаем только нужные поля и склеиваем путь до изображения
        image_path, question, answer = row[["image_path", "question", "answer"]]
        image_path = os.path.join(self.dataset_dir_path, image_path)

        return image_path, question, answer
