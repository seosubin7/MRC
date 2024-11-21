import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# 모델 및 토크나이저 로드
model_path = 'model3'
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_path)
max_seq_length = 1024
model.eval()

EOS_TOKEN = tokenizer.eos_token

# 새로운 프롬프트 형식
CustomDataset_train_prompt = """
컨텍스트: {context}
질문: {question}

세 가지 방법으로 답변을 도출하세요. 컨텍스트에 답이 있습니다. 
**답변은 '## 답변: ' 형식으로 작성하세요.**

1. 정보 추출: 
   - 핵심 단어: [질문에서 중요한 단어]
   - 관련 문장: [핵심 단어와 관련된 문장]
   - 요약: [관련 문장에서 답을 추출]
   - 답변 1: [첫 번째 답변 작성]

2. 맥락 기반 추론:
   - 주제 파악: [컨텍스트의 전반적인 흐름]
   - 연관 정보: [주제와 질문 연결]
   - 추론: [추론 결과 작성]
   - 답변 2: [두 번째 답변 작성]

3. 종합 및 선택:
   - 답변 비교: [답변 1과 2 비교]
   - 최종 답변: [가장 일관된 답변 선택]

## 답변:
"""

# test.json 파일 로드
print("data_test.json 파일 로드 중...")
with open('testc.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

def preprocess_function(examples):
    def clean_text(text):
        text = text.strip()
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9가-힣\s.,?!():~'\"']", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    inputs = [
        CustomDataset_train_prompt.format(
            context=clean_text(context),
            question=clean_text(question)
        ) 
        for context, question in zip(examples['context'], examples['question'])
    ]
    
    tokenized_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt"
    )

    labels = tokenized_inputs['input_ids'].clone()
    labels[tokenized_inputs['attention_mask'] == 0] = -100
    
    return {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': labels
    }

test_contexts = [item['context'] for item in test_data]
test_questions = [item['question'] for item in test_data]
test_answers = [item['answer'] for item in test_data]
preprocessed_data = preprocess_function({'context': test_contexts, 'question': test_questions})

# 단일 입력 예측 함수 정의
def predict_answer(preprocessed_inputs, attention_masks):
    results = []

    inputs = {
        'input_ids': preprocessed_inputs.to(model.device),
        'attention_mask': attention_masks.to(model.device)
    }
    with torch.no_grad():
        output = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=800,
            num_beams=4,
            repetition_penalty=1.8, 
            early_stopping =True
        )
    for o in output:
        answer = tokenizer.decode(o, skip_special_tokens=True)
        results.append(answer)
    
    return results

# 배치 사이즈 설정
batch_size = 2  # 원하는 배치 사이즈 설정

# 결과 저장을 위한 리스트
results = []

# tqdm을 이용하여 진행 상황 출력
total_questions = len(test_data)

# 모든 데이터를 처리 (배치 단위로)
for i in tqdm(range(0, total_questions, batch_size), desc="Processing Questions"):
    input_ids = preprocessed_data['input_ids'][i:i + batch_size]
    attention_mask = preprocessed_data['attention_mask'][i:i + batch_size]
    
    predicted_answers = predict_answer(input_ids, attention_mask)
    
    for j in range(len(predicted_answers)):
        question = test_questions[i + j]
        real_answer = test_answers[i + j]
    
        results.append({
            'question': question,
            'predicted_answer': predicted_answers[j] if predicted_answers else '',
            'real_answer': real_answer
        })
    
        print("Question:", question)
        print("Predicted Answer:", predicted_answers[j] if predicted_answers else '')
        print("Real Answer:", real_answer)

# 결과를 JSON 파일로 저장
with open('predicted_results_v3.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("예측 결과가 'predicted_results_v2.json' 파일에 저장되었습니다.")
