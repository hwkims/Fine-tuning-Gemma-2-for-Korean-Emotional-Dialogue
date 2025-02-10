import os
# Keras Backend 설정: JAX를 가장 먼저 설정해야 합니다.
os.environ["KERAS_BACKEND"] = "jax"
# 메모리 fragmentation 방지
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_nlp
import keras
from jax import config
import time

# JAX 환경설정, float32로 설정하여 Gemma 모델의 정확도 향상
config.update("jax_default_matmul_precision", "float32")

# 모델 ID 및 LoRA 설정
model_id = "gemma2_instruct_2b_en"  # 기본 모델 ID
lora_rank = 4
# Kaggle 모델 입력 경로 설정 (수정됨)
model_path = "/kaggle/input/gemma2-ko-dialogue-lora/keras/default/1/my_fine_tuned_gemma2_full_rank4.keras"
token_limit = 128 # 토큰 제한 설정

# 글로벌 시간 추적 변수
tick_start = 0

def tick():
    """시간 측정 시작."""
    global tick_start
    tick_start = time.time()

def tock():
    """시간 측정 종료 및 출력."""
    print(f"총 소요 시간: {time.time() - tick_start:.2f}s")

# Kaggle 모델 로드 (수정됨)
gemma_lm_loaded = keras.models.load_model(model_path)

# 텍스트 생성 함수 (수정됨)
def generate_text(prompt):
  tick() # 시간 측정 시작
  input_text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
  output = gemma_lm_loaded.generate(input_text, max_length=token_limit) # 모델 추론
  print(f"Generated text: {output}")
  tock() # 시간 측정 종료 및 출력

# 테스트 문장
test_prompt_1 = "오늘 기분이 너무 안좋아."
generate_text(test_prompt_1)

test_prompt_2 = "힘든 일이 있어서 위로가 필요해."
generate_text(test_prompt_2)
