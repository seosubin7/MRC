import json
from datasets import load_dataset

def format_and_save_datasets(datasets, splits, output_files, max_length=1024):
    """
    주어진 데이터셋과 split에 따라 데이터를 JSON 파일로 저장
    
    Args:
        datasets (list): 로드한 데이터셋들의 리스트
        splits (list): 저장할 데이터의 split 목록 (예: ['train', 'validation'])
        output_files (list): split별로 저장할 파일 이름 리스트
        max_length (int): context의 최대 허용 길이
    """
    for split, output_file in zip(splits, output_files):
        formatted_data = []

        # 각 데이터셋에서 split에 해당하는 데이터를 처리
        for dataset in datasets:
            if split in dataset:
                for item in dataset[split]:
                    context = item['context']
                    question = item['question']
                    # 'answers'는 여러 개의 답변이 있을 수 있으므로 첫 번째 텍스트를 사용
                    answer_text = item['answers']['text'][0] if 'answers' in item else ""

                    # context 길이 제한 적용
                    if len(context) <= max_length:
                        formatted_data.append({
                            'context': context,
                            'question': question,
                            'answer': answer_text
                        })

        # JSON 파일로 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)

        print(f"{split} 데이터셋 저장 완료: {output_file}")
        print(f"{len(formatted_data)}개의 항목이 저장되었습니다.")

# 데이터셋 로드
dataset_korquad = load_dataset("KorQuAD/squad_kor_v1")
dataset_klue = load_dataset("klue/klue", "mrc")

# 저장할 split 및 파일 이름 정의
splits = ['train', 'validation']
output_files = ['merged_train.json', 'merged_validation.json']


# 두 데이터셋을 리스트로 전달
format_and_save_datasets([dataset_korquad, dataset_klue], splits, output_files, max_length=1024)

