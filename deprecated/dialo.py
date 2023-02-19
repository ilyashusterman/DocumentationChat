import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

from storage import load_json


from transformers import AutoTokenizer
from torch.utils.data import Dataset


PATH = './ilya/blender_q_a'
PATH_ORIG = 'facebook/blenderbot-400M-distill'
# PATH_ORIG = './ilya/blender_q_a'
# PATH_ORIG = './ilya_blender/my-model'
# PATH_ORIG = 'Supiri/t5-base-conversation'
# PATH_ORIG = 'Supiri/t5-base-conversation'
# PATH_SAVE = './t5/my-model'
# PATH_SAVE = './ilya_blender/my-model'
# PATH = './ilya_blender/my-model'
# PATH = './t5/my-model'
# PATH = 'Supiri/t5-base-conversation'
# Define a custom Dataset class

class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=128, max_output_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the input and output texts from the data
        input_text = self.data[idx]["input"]
        output_text = self.data[idx]["output"]

        # Tokenize the input and output texts using the tokenizer
        input_tokens = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        output_tokens = self.tokenizer(
            output_text,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Return a dictionary of the tokenized input and output texts
        return {
            "input_ids": input_tokens["input_ids"][0],
            "attention_mask": input_tokens["attention_mask"][0],
            "decoder_input_ids": output_tokens["input_ids"][0][:-1],
            "decoder_attention_mask": output_tokens["attention_mask"][0][:-1],
            "labels": output_tokens["input_ids"][0][1:],
        }

# Instantiate the custom Dataset object


def get_question_answers_regex(obj):
    questions = re.findall(r'\nQuestion:.*\?\n', obj['text'])
    answers = re.findall(r'\nAnswer:.*\n', obj['text'])
    questions_answers = []
    if not answers or not questions:
        return questions_answers

    for q, a in zip(questions, answers):
        questions_answers.append({
            'question': q,
            'answer': a
        })
    return questions_answers


def get_train_data(data_objects):
    data = []
    for obj in data_objects:
        if 'questions_answers' in obj:
            for question_answer in obj['questions_answers']:
                data.append({
                    'output': question_answer['answer'],
                    'input': question_answer['question']
                })
        questions_answers = get_question_answers_regex(obj)
        for q_a in questions_answers:
            data.append({
                'output': q_a['answer'],
                'input': q_a['question']
            })
        data.append({
            'output': obj['text'],
            'input': obj['title']
        })
    return data


def train(num_train_epochs=2, path=PATH):
    data_objects = load_json('./questions_answers.json')
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "facebook/blenderbot-400M-distill")
    tokenizer = AutoTokenizer.from_pretrained(PATH_ORIG)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    train_data = get_train_data(data_objects)
    train_dataset = MyDataset(train_data, tokenizer)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        # evaluation_strategy="steps",
        num_train_epochs=num_train_epochs,
        learning_rate=2e-5,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=5,
        # warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        seed=42,
    )
    # Set up the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=val_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        tokenizer=tokenizer,
    )
    # Train the model
    trainer.train()
    # Save the model to the Hugging Face hub
    # trainer.save_model("./ilya_blender/my-model")
    trainer.save_model(PATH_SAVE)
    # tokenizer.save("./ilya/my-model")


def prompt(input_text, model, tokenizer):
    # Load the model from the Hugging Face hub


    # Define some input text
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate an output
    output_ids = model.generate(input_ids)

    # Decode the output to text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Print the output text
    print(output_text)


if __name__ == '__main__':
    # train()
    tokenizer = AutoTokenizer.from_pretrained(PATH_ORIG)
    model = AutoModelForSeq2SeqLM.from_pretrained(PATH)
    while True:
        question = input("\nPlease enter your question: ")
        if question == "quit":
            break
        else:
            prompt(question, model, tokenizer)


