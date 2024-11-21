import re
import json
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer, UnslothTrainingArguments
from accelerate import Accelerator
import matplotlib.pyplot as plt
from unsloth import is_bfloat16_supported

# Accelerator 초기화
accelerator = Accelerator()

max_seq_length = 1024
dtype = None
load_in_4bit = True

# 모델 및 토크나이저 로드
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-2b-it",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
    loftq_config=None,
)

# Self-consistency prompt 사용
CustomDataset_train_prompt = """
컨텍스트: {context}
질문: {question}

세 가지 방법으로 답변을 도출하세요. 
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

## 답변: {answer}
"""

# 데이터셋 로드
print("데이터셋 로드 중...")
dataset = load_dataset('json', data_files='trainc.json')

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
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
            question=clean_text(question),
            answer=clean_text(answer)
        ) + EOS_TOKEN
        for context, question, answer in zip(examples['context'], examples['question'], examples['answer'])
    ]
    
    # 텍스트를 토큰화하여 input_ids로 변환
    tokenized_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt"
    )

    # label에서 패딩 부분을 -100으로 설정
    labels = tokenized_inputs['input_ids'].clone()
    labels[tokenized_inputs['attention_mask'] == 0] = -100  # 패딩 부분을 -100으로 마스킹
    
    return {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': labels  # 손실 계산을 위한 label 설정
    }

# 데이터셋 전처리
processed_datasets = dataset.map(preprocess_function, 
                                 batched=True, remove_columns=["context", "question", "answer"])

# 데이터셋 분할 (80% 훈련, 20% 검증)
train_eval_split = processed_datasets['train'].train_test_split(test_size=0.13, seed=42)
train_dataset = train_eval_split['train']
eval_dataset = train_eval_split['test']

print(f"훈련 데이터셋 크기: {len(train_dataset)}")
print(f"검증 데이터셋 크기: {len(eval_dataset)}")

# Accelerator로 모델 준비 (테스트 데이터셋 없이)
model, train_dataset, eval_dataset = accelerator.prepare(model, train_dataset, eval_dataset)

# Trainer 설정 및 학습 실행
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=4,
    args=UnslothTrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        num_train_epochs=1,
        learning_rate=3e-5,
        embedding_learning_rate=3e-6,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=5,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs3",
    ),
)

trainer.train()

# 모델 저장
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("model3")
tokenizer.save_pretrained("model3")

# 학습 로그 저장
import json
with open('training_logs3.json', 'w') as f:
    json.dump(trainer.state.log_history, f)

# 학습 로그 시각화
logs = trainer.state.log_history
train_steps = [log["step"] for log in logs if "step" in log and "loss" in log]
train_losses = [log["loss"] for log in logs if "step" in log and "loss" in log]
eval_steps = [log["step"] for log in logs if "step" in log and "eval_loss" in log]
eval_losses = [log["eval_loss"] for log in logs if "step" in log and "eval_loss" in log]

plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_losses, label="Training Loss", marker='', markersize=3)
plt.plot(eval_steps, eval_losses, label="Evaluation Loss", marker='', markersize=3)
plt.xlabel("Steps")
plt.ylabel("Loss") 
plt.title("Training and Evaluation Loss over Steps")
plt.legend()  
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("training_eval_loss_sampled_v3.png", dpi=300, bbox_inches='tight')
plt.close()
