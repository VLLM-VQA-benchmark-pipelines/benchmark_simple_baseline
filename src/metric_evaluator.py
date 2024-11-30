import ast

import pandas as pd
import sacrebleu
from jiwer import cer, wer


class MetricEvaluator:
    def __init__(self, true_file, prediction_file):
        """
        Инициализация класса.
        :param true_file: путь к файлу с правильными ответами (CSV или TSV)
        :param prediction_file: путь к файлу с предсказаниями модели (CSV или TSV)
        """
        self.true_csv = self.read_file(true_file)
        self.pred_csv = self.read_file(prediction_file)
        self.validate_data()

    def read_file(self, file_path):
        """
        Чтение файла с определением разделителя (CSV или TSV).
        :param file_path: путь к файлу
        :return: DataFrame
        """
        try:
            return pd.read_csv(file_path, sep=";")  # Читаем как CSV
        except pd.errors.ParserError:
            return pd.read_csv(file_path, sep="\t")  # Если ошибка, читаем как TSV

    def validate_data(self):
        """
        Проверка совместимости данных.
        """
        # столбцы всегда будут не совпадать
        # if self.true_csv.columns.tolist() != self.pred_csv.columns.tolist():
        #     raise ValueError("Столбцы true_csv и pred_csv не совпадают.")

        if len(self.true_csv) != len(self.pred_csv):
            raise ValueError("Количество строк true_csv и pred_csv не совпадает.")

    def calculate_metrics_by_id(self):
        """
        Расчет метрик для каждого ID.
        :return: DataFrame с метриками для каждого ID.
        """
        # Создаем датафейм для результатов
        # Удаляем максимально избыточную информацию. Можно еще удалить "images_names"
        # такой колонки пока нет =)
        # result_df = self.true_csv.drop(columns='answear_bbox')
        result_df = self.true_csv

        # Создаем хранилища для метрик
        wer_error_list = []
        cer_error_list = []
        bleu_score_list = []

        for row in range(len(self.true_csv)):
            # Преобразуем строку в список. Пример: "['Ответ 1', 'Ответ 2']" -> ['Ответ 1', 'Ответ 2']
            Y_true = self.true_csv["answer"][row]
            y_pred = self.pred_csv["answer"][row]

            # Проверка на кол-во ответов
            # if len(Y_true) != len(y_pred):
            #     pass  # Что-то надо придумать или не надо...

            # Вычисляем метрику WER
            wer_error = wer(Y_true, y_pred)
            wer_error_list.append(wer_error)

            # Вычисляем метрику CER
            cer_error = cer(Y_true, y_pred)
            cer_error_list.append(cer_error)

            # Вычисляем метрику BLEU
            bleu_score = sacrebleu.corpus_bleu(Y_true, [y_pred]).score
            bleu_score_list.append(bleu_score)

        # Дополняем наш результирующий датафрейм метриками
        metrics = {
            "pred_answers": self.pred_csv["answer"],
            "wer_error": wer_error_list,
            "cer_error": cer_error_list,
            "bleu_score": bleu_score_list,
        }
        metrics = pd.DataFrame(metrics)
        result_df = pd.concat([result_df, metrics], axis=1)

        return result_df

    def calculate_metrics_by_doc_type(self, df):
        """
        Расчет метрик для каждого типа документа.
        :param df: DataFrame из метода calculate_metrics_by_id
        :return: DataFrame с метриками для каждого типа документа.
        """
        # Создаем список из всех уникальных типов документов
        doc_types = df["doc class"].unique()

        # Создаем хранилища для метрик 
        results = {
        "doc_class": [],
        "wer_error": [],
        "cer_error": [],
        "bleu_score": []
        }

        for doc_type in doc_types:
            # Фильтрация по типу документа
            subset = df[df["doc class"] == doc_type]
            
            # Подсчёт общих ошибок и слов для WER
            total_wer_errors = subset["wer_errors"].sum()
            total_words = subset["word_count"].sum()
            wer_error = total_wer_errors / total_words if total_words > 0 else 0
            
            # Подсчёт общих ошибок и символов для CER
            total_cer_errors = subset["cer_errors"].sum()
            total_chars = subset["char_count"].sum()
            cer_error = total_cer_errors / total_chars if total_chars > 0 else 0

            # Средний BLEU
            bleu_score = subset["bleu_score"].mean()

            # Добавление метрик в результаты
            results["doc_class"].append(doc_type)
            results["wer_error"].append(wer_error)
            results["cer_error"].append(cer_error)
            results["bleu_score"].append(bleu_score)

        return pd.DataFrame(results)

    def group_by_doc_question(self, df): 
        """
        Группировка по типу документа и типу вопроса.
        :param df: pandas.DataFrame - исходный датафрейм
        :return: pandas.DataFrame - сгруппированный датафрейм с метриками
        """
        # Группировка с подсчетом общих ошибок и токенов
        grouped = df.groupby(['doc_class', 'question_type']).apply(lambda group: pd.Series({
            'wer_error': group['wer_errors'].sum() / group['word_count'].sum() if group['word_count'].sum() > 0 else 0,
            'cer_error': group['cer_errors'].sum() / group['char_count'].sum() if group['char_count'].sum() > 0 else 0,
            'bleu_score': group['bleu_score'].mean()  # BLEU можно усреднить напрямую
        })).reset_index()

        return grouped