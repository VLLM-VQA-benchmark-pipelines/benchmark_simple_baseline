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
        self.dataframe = pd.read_csv(annot_path, sep=";")

    def __iter__(self):
        return self

    def __next__(self):
        """Возвращаем склееный путь до изображения, вопрос и ответ на него"""
        # инкремент для индекса
        row_index = self.row_index
        self.row_index += 1

        # получаем только нужные поля и склеиваем путь до изображения
        row = self.dataframe.loc[row_index]
        image_path, question, answer = row[['image_path', 'question', 'answer']]
        image_path = os.path.join(self.dataset_dir_path, image_path)
        
        return image_path, question, answer
