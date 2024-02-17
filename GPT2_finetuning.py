import pandas as pd
import numpy as np
import re
import os

def read_txt(file_path):
    with open(file_path, "rb") as file:
        text = file.read()
def read_documents_from_directory(directory):
    combined_text = ""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
    if filename.endswith(".txt"):
            combined_text += read_txt(file_path)
    return combined_text
'''
train_directory = './data/shakespeare'
text_data = read_documents_from_directory(train_directory)
text_data = re.sub(r'\n+', '\n', text_data).strip()

data_file = train_directory+"/myself_data.txt"
with open(data_file) as f:
    f.write(text_data)
'''

# 前面的代码属于数据准备和和预处理阶段
# 如果有已经处理好了的数据，我们可以直接采用

from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset

def load_data_collator(tokenizer, mlm=False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
    )
    return data_collator

def train(train_file_path, model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.save_pretrained(output_dir)

    trainning_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
    )

    trainer = Trainer(
        model=model,
        args=trainning_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()


train_file_path = "./data/shakespeare/input.txt"
model_name='gpt2-large'
output_dir = './model/gpt2_finetuing'
overwrite_output_dir = False
per_device_train_batch_size=8
num_train_epochs = 50.0
save_steps = 5000

train(train_file_path,model_name,output_dir,overwrite_output_dir,per_device_train_batch_size,num_train_epochs,save_steps)

