import os
import json
import time

# Set the backend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_nlp
import keras
import matplotlib.pyplot as plt

# --- Training Configurations ---
token_limit = 256
num_data_limit = None  # Train on all data, or set a limit for testing
lora_name = "korean"  # Name for your LoRA weights
lora_rank = 2        # LoRA rank
lr_value = 1e-4
train_epoch = 2
model_id = "gemma2_instruct_2b_en"  # Base Gemma model ID


# --- Load Model ---
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
gemma_lm.summary()


# --- Helper Functions ---
def tick():
    global tick_start
    tick_start = time.time()

def tock():
    print(f"TOTAL TIME ELAPSED: {time.time() - tick_start:.2f}s")

def text_gen(prompt):
    tick()
    input_text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    output = gemma_lm.generate(input_text, max_length=token_limit)
    print("\nGemma output:")
    print(output)
    tock()


# --- Load Dataset ---
tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)

def load_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

train_data_raw = load_data_from_json("/kaggle/input/training/Training.json") #  *CHANGE PATH IF NEEDED*
validation_data_raw = load_data_from_json("/kaggle/input/validation/Validation.json")  # *CHANGE PATH IF NEEDED*

def preprocess_dialogue(item):
    """Extracts and formats dialogue turns."""
    dialogue = item['talk']['content']
    prompt = ""
    response = ""
    history = ""
    for i in range(1, 10):  # Assuming max 9 turns of context
        human_key = f"HS0{i}" if i < 10 else f"HS{i}"
        system_key = f"SS0{i}" if i < 10 else f"SS{i}"

        if human_key in dialogue and dialogue[human_key]:
            history += f"<start_of_turn>user\n{dialogue[human_key]}<end_of_turn>\n"
            if system_key in dialogue and dialogue[system_key]:
                response = dialogue[system_key]  # Capture the *next* system response
            else:
                # If no system response, then this human turn is the prompt itself.
                prompt = dialogue[human_key]
                return prompt, None # No target response to train on

        elif system_key in dialogue and dialogue[system_key]:
             history += f"<start_of_turn>model\n{dialogue[system_key]}<end_of_turn>\n"


    if history and response: # We have a history and a valid response
        return history.strip(), response.strip()  # Return history and the response

    return None, None  # No valid training example



train_texts = []
for item in train_data_raw:
    prompt, response = preprocess_dialogue(item)
    if prompt and response:
        full_text = f"{prompt}<start_of_turn>model\n{response}<end_of_turn>"
        # Tokenize and check length *before* adding to the list
        if len(tokenizer(full_text)) < token_limit:
             train_texts.append(full_text)

if num_data_limit: # Optional data limiting
    train_texts = train_texts[:num_data_limit]


print(f"Number of training samples: {len(train_texts)}")
print("Example training sample:")
print(train_texts[0])



# --- LoRA Fine-tuning ---

# Enable LoRA for the model and set the LoRA rank.
gemma_lm.backbone.enable_lora(rank=lora_rank)
gemma_lm.summary()

# Limit the input sequence length (to control memory usage).
gemma_lm.preprocessor.sequence_length = token_limit

# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_value,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


# Custom Callback for saving LoRA weights and evaluation
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        model_name = f"/kaggle/working/{lora_name}_{lora_rank}_epoch{epoch+1}.lora.h5"
        gemma_lm.backbone.save_lora_weights(model_name)
        print(f"Saved LoRA weights to {model_name}")

        # Evaluate on a sample from the validation set
        if validation_data_raw:
            sample_item = validation_data_raw[0]  # Take the first item
            eval_prompt, _ = preprocess_dialogue(sample_item)
            if eval_prompt:
                text_gen(eval_prompt)
            else:
                print("Could not create evaluation prompt from validation sample.")


history = gemma_lm.fit(train_texts, epochs=train_epoch, batch_size=1, callbacks=[CustomCallback()])

# Plot training loss
plt.plot(history.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()


# --- Load and Test LoRA (Optional, but good practice) ---
print("\nLoading LoRA weights...")
gemma_lm_loaded = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
gemma_lm_loaded.backbone.enable_lora(rank=lora_rank)
# Load the weights from the *last* epoch
last_epoch = train_epoch
lora_weights_path = f"/kaggle/working/{lora_name}_{lora_rank}_epoch{last_epoch}.lora.h5"
gemma_lm_loaded.backbone.load_lora_weights(lora_weights_path)
print(f"LoRA weights loaded from {lora_weights_path}")

# Try the loaded model
print("\nTesting the loaded model:")
if validation_data_raw:  # Use validation data for a quick test
    sample_item = validation_data_raw[0]
    test_prompt, _ = preprocess_dialogue(sample_item)
    if test_prompt:
        text_gen(test_prompt)
    else:
        print("Could not create test prompt from validation sample.")
