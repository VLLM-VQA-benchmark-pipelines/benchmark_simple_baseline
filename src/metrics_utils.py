from jiwer import wer, cer
import sacrebleu


def calculate_metrics(answers_by_image, model_answers):
    # Вычисляем метрики WER и CER
    wer_error = wer(answers_by_image, model_answers)
    cer_error = cer(answers_by_image, model_answers)
    # Вычисляем BLEU
    bleu_score = sacrebleu.corpus_bleu(answers_by_image, [model_answers])
    benchmark_results = {
        "wer_error": wer_error,
        "cer_error": cer_error,
        "BLEU score": bleu_score.score,
    }
    
    return benchmark_results
