import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, \
    TrainingArguments, pipeline, T5ForConditionalGeneration, T5Tokenizer
import json

from storage.json import load_q_a

# Load and preprocess the dataset
# data = load_q_a()
#
# questions = [item['question'] for item in data]
# answers = [item['answer'] for item in data]
#
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# tokenizer.pad_token = tokenizer.eos_token
# encoded_questions = tokenizer(questions, truncation=True, max_length=128, return_overflowing_tokens=True,
#                             return_length=True, return_tensors='pt')
# encoded_answers = tokenizer(answers,return_overflowing_tokens=True,
#                             return_length=True, truncation=True, max_length=128, return_tensors='pt')

# Fine-tune the GPT-2 model
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
#     save_total_limit=3,
#     learning_rate=1e-4
# )
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=list(zip(encoded_questions['input_ids'], encoded_answers['input_ids'])),
#     data_collator=lambda data: {'input_ids': torch.stack([x[0] for x in data]), 'attention_mask': torch.stack([x[1] for x in data]), 'labels': torch.stack([x[1] for x in data])},
# )
# trainer.train()

# Generate answers
# generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
# prompt = 'What is the capital of France?'
# answer = generator(prompt, max_length=256, do_sample=True)[0]['generated_text']
# print(answer)


def generate_summary(question, title, answer):
    # Concatenate the question and title to form the input prompt
    prompt = f"question: {question}\narticle title: {title}\narticle body: {answer}"

    # Load the T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    # Encode the prompt and generate the summary
    inputs = tokenizer(prompt, padding='max_length', truncation=True,
                       max_length=512, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], max_length=700,
                                 num_beams=4,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

question = "how do i integrate google analytics?"
title = "integrate google analytic ?"
answer = "Google Ads (AdWords)—re-engagement\nGoogle Ads (AdWords)—cost and ad revenue\nGoogle Ads (AdWords)—FAQ and discrepancies (this article)\nFor iOS 14-related campaigns see Google Ads SKAdNetwork interoperability with AppsFlyer.\n\n\nGoogle Ads FAQ\n\n\nWhy am I not seeing clicks from Google Ads?\n\nAggregate clicks and impressions data from Google Ads is collected once you have authenticated cost, clicks and impressions collection"

if __name__ == '__main__':

    summary = generate_summary(question, title, answer)
    print(summary)