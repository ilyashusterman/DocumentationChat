import re

import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, \
    DistilBertForQuestionAnswering, Trainer, TrainingArguments, \
    DistilBertForMaskedLM, DistilBertModel, pipeline, Conversation, \
    AutoTokenizer, GPT2LMHeadModel, TextDataset, \
    DataCollatorForLanguageModeling, AutoModelWithLMHead, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from questions.prompt import prompt, prompt_conversation


def get_data():
    # Prepare training data
    return [
                'The merge function in Pandas is used to combine two or more DataFrames based on the values of one or more columns.',
                'To merge two DataFrames in Pandas, use the merge function and specify the two DataFrames as the first two arguments. You can specify the column(s) to merge on using the `on` parameter.',
                'There are four types of merge in Pandas: inner join, left join, right join, and outer join. You can specify the type of merge using the `how` parameter.']



def build_text_files(data_json, dest_path):
    f = open(dest_path, 'w')
    data = ''
    for texts in data_json:
        for answer in texts:
            # summary = str(texts['answer']).strip()
            summary = re.sub(r"\s", " ", answer)
            data += summary + "  "
    f.write(data)

def load_dataset(train_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,data_collator

def train(model, tokenizer, save_path):
    data = get_data()
    train_path = './train_dataset.txt'
    build_text_files(data, train_path)
    train_dataset, data_collator = load_dataset(train_path, tokenizer)

    training_args = TrainingArguments(
        output_dir="./gpt2-results",  # The output directory
        overwrite_output_dir=True,
        # overwrite the content of the output directory
        num_train_epochs=3,  # number of training epochs
        per_device_train_batch_size=32,  # batch size for training
        per_device_eval_batch_size=64,  # batch size for evaluation
        eval_steps=400,  # Number of update steps between two evaluations.
        save_steps=800,  # after # steps model is saved
        warmup_steps=500,
        # number of warmup steps for learning rate scheduler
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        # prediction_loss_only=True,
    )
    trainer.train()
    trainer.save_model(save_path)



def answer_question(question, chatbot):
    # Tokenize the question
    # conversation = Conversation(question)
    output = chatbot(question)

    return output

#
if __name__ == '__main__':
    SAVE_PATH = './models/gpt'
    # checkpoint_model = 'distilbert-base-uncased'
    checkpoint_model = SAVE_PATH
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side='left')
    model = AutoModelWithLMHead.from_pretrained(SAVE_PATH)
    # train(model, tokenizer, save_path=SAVE_PATH)
    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # model = DistilBertModel.from_pretrained(checkpoint_model)
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # train(model, tokenizer, save_path=SAVE_PATH)
    chatbot= pipeline('conversational', model=model, tokenizer=tokenizer)
    prompt_conversation(
        func=answer_question,
        chatbot=chatbot,
    )
