 
![image](https://github.com/user-attachments/assets/ea2613d0-9c8d-4781-9a9a-964b5f7191a6)
# 한국어 감성 대화를 위한 Gemma2 기반 모델 (Gemma2 for Korean Empathetic Dialogue)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/hwkims/fork-of-gemma2-for-korean-mental-wellbeing-infere)

## 소개 (Introduction)

본 레포지토리는 Google의 Gemma2 (2B 파라미터) 모델을 기반으로 한국어 감성 대화 지원을 위한 챗봇을 구현하는 프로젝트입니다. [Google - Unlock Global Communication with Gemma](https://www.kaggle.com/competitions/google-gemma/overview) Kaggle Competition에 참여하면서 개발되었습니다. [gemma2-ko-dialogue-lora](https://www.kaggle.com/models/hyun-woo-kim/gemma2-ko-dialogue-lora) 모델(LoRA)은 *AI Hub의 감성 대화 말뭉치 데이터를 사용하여 Gemma2 모델을 fine-tuning 한 결과*입니다. 공감적인 대화 능력을 향상시키기 위해 노력했습니다.

## 모델 (Model)

*   **기반 모델 (Base Model):** Gemma2 (2B)
*   **LoRA 모델:**  [gemma2-ko-dialogue-lora](https://www.kaggle.com/models/hyun-woo-kim/gemma2-ko-dialogue-lora) (Kaggle Models에서 다운로드 가능)
    *   *LoRA (Low-Rank Adaptation) 기법을 사용하여 fine-tuning 되었습니다.*
    *   *Fine-tuning 과정은 다음 Kaggle 노트북에서 확인할 수 있습니다: [Gemma2 2B LoRA + ED (AI Hub) Fine-tuning](https://www.kaggle.com/code/hwkims/gemma2-2b-lora-ed-ai-hub-fine-tuning)*
    *   [선택사항, 다른 LoRA 모델이 있다면 추가] [gemma2-2b-ko-empathetic-lora](https://www.kaggle.com/models/hyun-woo-kim/gemma2-2b-ko-empathetic-lora) ( *이 모델도 사용한다면, 다운로드 및 사용법을 README에 추가해야 합니다.*)

## 파인튜닝 (Fine-tuning)

[gemma2-ko-dialogue-lora](https://www.kaggle.com/models/hyun-woo-kim/gemma2-ko-dialogue-lora) 모델은 다음의 과정을 거쳐 파인튜닝 되었습니다.  자세한 내용은 [Fine-tuning Kaggle Notebook](https://www.kaggle.com/code/hwkims/gemma2-2b-lora-ed-ai-hub-fine-tuning)에서 확인할 수 있습니다.

1.  **데이터셋 준비:**
    *   AI Hub의 [감성 대화 말뭉치](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=86)를 사용했습니다.
    *   데이터는 JSON 형식으로 제공되며,  `Training.json`과 `Validation.json` 파일을 사용했습니다.
    *   데이터 전처리 과정:
        *   대화 내용을 추출하여 사용자 발화(prompt)와 모델 응답(response)으로 구성했습니다.
        *   최대 4턴의 대화 기록(history)을 포함하여 문맥을 파악할 수 있도록 했습니다.
        *   `keras_nlp`의 `GemmaTokenizer`를 사용하여 텍스트를 토큰화했습니다.
        *   최대 토큰 길이를 256으로 제한하여 메모리 사용량을 관리했습니다.

2.  **LoRA 적용:**
    *   `keras_nlp`의 `GemmaCausalLM` 모델을 사용했습니다.
    *   `enable_lora(rank=2)`를 호출하여 LoRA를 적용했습니다. (LoRA rank는 2로 설정)
    *   LoRA를 사용하면 학습 파라미터 수를 줄여 효율적인 파인튜닝이 가능합니다.

3.  **모델 컴파일:**
    *   `AdamW` 옵티마이저를 사용했습니다. (learning_rate=1e-4, weight_decay=0.01)
    *   `SparseCategoricalCrossentropy` 손실 함수를 사용했습니다. (from_logits=True)
    *   `SparseCategoricalAccuracy`를 평가 지표로 사용했습니다.
    *   `exclude_from_weight_decay(var_names=["bias", "scale"])`을 통해 layernorm과 bias에는 weight decay가 적용되지 않도록 했습니다.

4.  **학습:**
    *   `fit()` 메서드를 사용하여 모델을 학습했습니다.
    *   배치 크기(batch size)는 1로 설정했습니다. ( *Kaggle 환경 및 GPU 메모리 제약 고려* )
    *   총 2 epoch 동안 학습했습니다.
    *   `CustomCallback`을 사용하여 매 epoch마다 LoRA 가중치를 저장했습니다. (`/kaggle/working/{lora_name}_{lora_rank}_epoch{epoch+1}.lora.h5`)
    *    매 epoch 마다 검증 데이터의 첫 번째 샘플로 추론을 수행하여 모델의 성능 변화를 확인했습니다.

5.  **학습 결과:**
    *  [Fine-tuning Kaggle Notebook](https://www.kaggle.com/code/hwkims/gemma2-2b-lora-ed-ai-hub-fine-tuning) 실행 결과 생성된 LoRA 가중치 파일을 다운로드 받아 사용할 수 있습니다.
    *  [그래프 추가]: 학습 과정에서의 loss 변화를 보여주는 그래프를 첨부하면 좋습니다. (Kaggle 노트북의 Matplotlib 그래프를 이미지로 저장하여 첨부)

## 추론 노트북 (Inference Notebook)

`inference_notebook.ipynb` 에는 모델을 로드하고, 사용자 입력을 받아 응답을 생성하는 추론 코드가 포함되어 있습니다.
* **Kaggle Inference Notebook:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/hwkims/fork-of-gemma2-for-korean-mental-wellbeing-infere)

### 노트북 실행 환경 설정 (Setup)

1.  **필수 패키지 설치:** `requirements.txt` 파일에 명시된 패키지들을 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```
    ([`requirements.txt` 생성 방법](#requirements-txt-생성-선택) 참고)

2.  **모델 다운로드:**
    *   **Kaggle API 사용 (권장):** Kaggle API를 이용하여 `model/` 폴더에 [gemma2-ko-dialogue-lora](https://www.kaggle.com/models/hyun-woo-kim/gemma2-ko-dialogue-lora) 모델을 다운로드 받습니다. (Kaggle API 사용법 링크 또는 설명 추가).
        ```bash
        kaggle models download -m hyun-woo-kim/gemma2-ko-dialogue-lora -p model/gemma2-ko-dialogue-lora
        ```
    *   **수동 다운로드:** Kaggle Models 페이지에서 직접 모델 파일을 다운로드 받아 `model/gemma2-ko-dialogue-lora` 폴더에 저장합니다.
      *  또는 파인튜닝 노트북 실행으로 생성된 LoRA weight 파일(`*.lora.h5`)을 `model/` 폴더에 넣습니다.

### 추론 실행 (Inference)

`inference_notebook.ipynb` 파일을 열고, 노트북의 셀들을 순서대로 실행합니다. 또는 Kaggle Inference Notebook을 통해 실행할 수 있습니다.

## 데이터셋 (Dataset)

본 모델은 AI Hub의 **감성 대화 말뭉치** 데이터를 사용하여 fine-tuning 되었습니다.

*   **데이터셋 이름:** 감성 대화 말뭉치
*   **AI Hub 링크:** [https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=86](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=86)
*   **데이터 설명:**
    *   한국어 감성 대화 데이터셋으로, 다양한 감정 상황에서의 대화 텍스트와 음성 데이터로 구성되어 있습니다.
    *   우울증 예방 및 감성 챗봇 개발 등에 활용될 수 있습니다.
*   **데이터 다운로드:**
    *   AI Hub 웹사이트에서 회원 가입 및 로그인 후 다운로드 가능합니다. (내국인만 가능)
    *   [AI Hub 데이터 다운로드 페이지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=86)에서 "다운로드" 버튼을 클릭하여 다운로드 받을 수 있습니다.
    *   *데이터 용량이 크므로(20.35MB), 다운로드 시 주의하시기 바랍니다.*

## 참고 자료 (References)

*   [Gemma2 모델 설명](링크)  (Gemma2 관련 공식 문서나 블로그 글 링크)
*   [LoRA (Low-Rank Adaptation) 설명](링크) (LoRA 논문이나 설명 자료 링크)
*   [Kaggle Competition: Google - Unlock Global Communication with Gemma](https://www.kaggle.com/competitions/google-gemma/overview)
*  [Fine-tuning Kaggle Notebook](https://www.kaggle.com/code/hwkims/gemma2-2b-lora-ed-ai-hub-fine-tuning)
*  [Inference Kaggle Notebook](https://www.kaggle.com/code/hwkims/fork-of-gemma2-for-korean-mental-wellbeing-infere)

## 기여 (Contribution)

[기여 방법에 대한 설명. Pull Request 가이드라인 등.  예시: ]
Pull requests are welcome.  For major changes, please open an issue first to discuss what you would like to change.

## 라이선스 (License)

본 프로젝트는 [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0)를 따릅니다. AI Hub의 "감성 대화 말뭉치" 데이터는 별도의 이용 약관 및 라이선스를 따르므로, 사용 시 주의하시기 바랍니다.

## (선택) `requirements.txt` 생성

```bash
pip freeze > requirements.txt
