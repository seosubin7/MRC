import json
import nltk
from nltk.tokenize import word_tokenize
import re
import string
from nltk.translate.meteor_score import meteor_score  # METEOR 점수 계산 모듈 추가
from rouge_score import rouge_scorer  # ROUGE-L 점수 계산 모듈 추가

nltk.download('punkt')

def load_predictions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
   
    return predictions

def normalize_answer(s):
    """소문자화, 구두점 제거, 불필요한 공백 제거 등을 수행"""
    
    def remove_punctuation(text):
        # set(string.punctuation)에 '*' 문자를 포함시켜 제거
        return ''.join(ch for ch in text if ch not in set(string.punctuation + '*'))

    def lower(text):
        return text.lower()

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_newlines(text):
        return text.replace('\\n', ' ')  # 문자열로 포함된 \n을 공백으로 대체

    def remove_letter_dot(text):
        # 'A.', 'B.'와 같은 알파벳과 점을 제거
        return re.sub(r'\b[A-Za-z]\.\s?', '', text)

    def remove_empty_quotes(text):
        # 빈 따옴표('') 제거
        return text.replace("''", '')

    def filter_characters(text):
        # 알파벳, 숫자, 한글, 공백, 일부 구두점(.,?!())만 남기고 제거
        return re.sub(r"[^a-zA-Z0-9가-힣\s.,?!()]", "", text)

    def remove_empty_parentheses(text):
        # 빈 괄호 () 제거
        return re.sub(r'\(\s*\)', '', text)

    return white_space_fix(remove_articles(remove_punctuation(lower(remove_letter_dot
            (remove_empty_parentheses(remove_empty_quotes(remove_newlines(filter_characters(s)))))))))

def calculate_f1(prediction, ground_truth):
    prediction_tokens = word_tokenize(normalize_answer(prediction))
    ground_truth_tokens = word_tokenize(normalize_answer(ground_truth))
    
    common = set(prediction_tokens) & set(ground_truth_tokens)
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0
    
    precision = len(common) / len(prediction_tokens)
    recall = len(common) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        return 0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def calculate_em(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def calculate_meteor(prediction, ground_truth):
    # METEOR 점수 계산을 위해 토큰화된 리스트 사용
    prediction_tokens = word_tokenize(normalize_answer(prediction))
    ground_truth_tokens = word_tokenize(normalize_answer(ground_truth))
    return meteor_score([ground_truth_tokens], prediction_tokens)

def calculate_rouge_l(prediction, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return scores['rougeL'].fmeasure

def main():
    predictions = load_predictions('gemma_result_modified.json')
    
    predicted_answers = [p['predicted_answer'] for p in predictions]
    actual_answers = [p['real_answer'] for p in predictions]
    
    # Exact Match 계산
    em_scores = [calculate_em(pred, actual) for pred, actual in zip(predicted_answers, actual_answers)]
    em_score = sum(em_scores) / len(em_scores)
    
    # F1 Score 계산
    f1_scores = [calculate_f1(pred, actual) for pred, actual in zip(predicted_answers, actual_answers)]
    avg_f1_score = sum(f1_scores) / len(f1_scores)

    # METEOR Score 계산
    meteor_scores = [calculate_meteor(pred, actual) for pred, actual in zip(predicted_answers, actual_answers)]
    avg_meteor_score = sum(meteor_scores) / len(meteor_scores)
    
    # ROUGE-L Score 계산
    rouge_l_scores = [calculate_rouge_l(pred, actual) for pred, actual in zip(predicted_answers, actual_answers)]
    avg_rouge_l_score = sum(rouge_l_scores) / len(rouge_l_scores)

    # 결과 출력
    print(f"Exact Match Score: {em_score * 100:.4f}")  # 100점 만점으로 표시
    print(f"Average F1 Score: {avg_f1_score * 100:.4f}")  # 100점 만점으로 표시
    print(f"Average METEOR Score: {avg_meteor_score * 100:.4f}")  # 100점 만점으로 표시
    print(f"Average ROUGE-L Score: {avg_rouge_l_score * 100:.4f}")  # 100점 만점으로 표시

if __name__ == "__main__":
    main()
