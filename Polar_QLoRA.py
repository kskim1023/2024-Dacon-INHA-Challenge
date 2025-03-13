# import warnings
# warnings.filterwarnings('ignore')

import json
import torch
import logging
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, pipeline
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
from accelerate import Accelerator
from collections import Counter

MODEL_NAME = 'x2bee/POLAR-14B-v0.5'

#LOG --------------------------------------------------------------------------
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

file_handler = logging.FileHandler('./results/POLAR/training2.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
#--------------------------------------------------------------------------

#DATASET --------------------------------------------------------------------------
logger.info('Loading dataset...')
with open('train_chunk.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert data into a dataset object
dataset = Dataset.from_list(data)
train_val_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']
#--------------------------------------------------------------------------

#TOKEN -------------------------------------------------------------------------
logger.info('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = 'right'  # Ensure padding side is right

def preprocess_function(examples):
    inputs = [f"너는 주어진 한국 경제 기사 Context를 토대로 Question에 대답해. Question에 대한 답변은 최대한 간결하게 1 단어로 답변해. Context: {context} Question: {question}\nAnswer: {answer}" 
              for context, question, answer in zip(examples['context'], examples['question'], examples['answer'])]
    return tokenizer(inputs, truncation=True, padding='max_length', max_length=1024)

# Tokenizing dataset
logger.info('Tokenizing dataset...')
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)
#--------------------------------------------------------------------------

#CONFIGS --------------------------------------------------------------------------
# Quantization config
logger.info('Setting up quantization configuration...')
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4', 
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

logger.info('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    quantization_config=quant_config,
    device_map='auto',
    trust_remote_code=True
    #rope_scaling = {"type": "dynamic", "factor": 2}
)

model.config.use_cache=False
model.config.pretraining_tp=1

logger.info('Configuring LoRA...')
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16,
    target_modules=['v_proj', 'k_proj', 'o_proj', 'q_proj'], #'up_proj', 'down_proj', 'gate_proj'
    bias="none",
    task_type="CAUSAL_LM"
)

logger.info('Applying LoRA to the model...')
model = get_peft_model(model, lora_config)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results/POLAR',
    eval_strategy='epoch',
    save_strategy='epoch',
    optim="paged_adamw_8bit",
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=20000,
    fp16=True,
    logging_dir='./results/POLAR/logs',
    save_total_limit=1,
    load_best_model_at_end=True
)

# Initializing Accelerator
logger.info('Initializing Accelerator...')
accelerator = Accelerator(mixed_precision='fp16')

# Prepare the model, tokenizer, and data with Accelerator
model, tokenizer, tokenized_train_dataset, tokenized_val_dataset = accelerator.prepare(
    model, tokenizer, tokenized_train_dataset, tokenized_val_dataset
)

# Initializing pipeline
logger.info('Initializing pipeline...')
qa_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer)

def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()

    prediction_Char = [char for token in prediction_tokens for char in token]
    ground_truth_Char = [char for token in ground_truth_tokens for char in token]

    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

def evaluate(ground_truth_df, predictions_df):
    predictions = dict(zip(predictions_df['question'], predictions_df['answer']))
    f1 = total = 0

    for index, row in ground_truth_df.iterrows():
        question_text = row['question']
        ground_truths = row['answer']
        total += 1
        if question_text not in predictions:
            continue
        prediction = predictions[question_text]
        f1 += f1_score(prediction, ground_truths)

    f1 = 100.0 * f1 / total
    return {'f1': f1}

def generate_response(question_prompt):
    response = qa_pipeline(question_prompt, max_new_tokens=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
    if "Answer:" in response:
        response = response.split("Answer:", 1)[1][:20]

        if "Que" in response:
            response = response.split("Que", 1)[0]
        if "⊙" in response:
            response = response.split("⊙", 1)[0]
        if "Con" in response:
            response = response.split("Con", 1)[0]
    return response

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = tokenizer.batch_decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)
    references = tokenizer.batch_decode(labels, skip_special_tokens=True)

    ground_truth_df = pd.DataFrame({'question': val_dataset['question'], 'answer': val_dataset['answer']})
    predictions_df = pd.DataFrame({'question': val_dataset['question'], 'answer': predictions})

    results = evaluate(ground_truth_df, predictions_df)
    return results

logger.info('Initializing SFTTrainer...')
trainer = SFTTrainer(
    model=model,
    args=training_args,
    peft_config=lora_config,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

logger.info('Starting training...')
trainer.train()

logger.info('Saving the model...')
trainer.save_model('./model/POLAR')

logger.info('Training complete.')
