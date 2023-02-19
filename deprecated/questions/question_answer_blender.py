import math

from datasets import Dataset, load_dataset
from transformers import BlenderbotSmallTokenizer, \
    DataCollatorForLanguageModeling, pipeline, \
    BlenderbotForConditionalGeneration, AutoTokenizer, \
    BlenderbotSmallForCausalLM
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, BlenderbotForCausalLM

from questions.generate_questions import clean_text
from questions.prompt import prompt_conversation, prompt
from storage import get_data, load_json, save_json



def group_texts(examples, block_size=128):
    concatenated_examples = {k: sum(examples[k], []) for k in
                             examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in
            range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def train(model, tokenizer, save_path):
    data_path = './apps_flyers.json'
    data_path_test = './apps_flyers_test.json'
    questions_answers = load_json('./process_q_a_2.json')
    data = get_data()
    text_data = [{'text': clean_text(d['text'])} for d in data]
    # text_data = text_data + [{'text': f"{a['question']} {a['answer']}"} for a
    #                          in questions_answers]
    save_json(text_data, data_path)
    save_json(text_data[:100], data_path_test)
    lm_dataset = load_dataset_lm(data_path, tokenizer)
    lm_dataset_test = load_dataset_lm(data_path_test, tokenizer)


    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=False)

    training_args = TrainingArguments(
        output_dir=f'{save_path}/results/',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        eval_dataset=lm_dataset_test,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(
        save_path
    )
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    # trainer.save_state(save_path)


def load_dataset_lm(data_path, tokenizer):
    dataset = load_dataset("json", data_files=f'.{data_path}', split="train")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=['text'],
    )
    # lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)
    return tokenized_dataset


def generated_text(prompt, chatbot):
    result = chatbot(prompt)
    print()
    return result

def question(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for r in result:
        print(r)
    return result

def question_blender(prompt, model, tokenizer):
    # inputs = tokenizerr([prompt], return_tensors="pt")
    # # reply_ids = model.generate(**inputs)
    # reply_ids = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50,
    #                    top_p=0.95)
    # result = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(result)
    return result

if __name__ == '__main__':
    # model_checkpoint = 'facebook/blenderbot-400M-distill'
    model_checkpoint_token = 'facebook/blenderbot-400M-distill'
    # model_checkpoint = "distilgpt2"
    # model_checkpoint_token = "distilgpt2"
    # model_checkpoint_token = "facebook/blenderbot_small-90M"
    # model_checkpoint = "facebook/blenderbot_small-90M"
    # model_checkpoint_token = "learned_doc_appsflyers/model"
    # model_checkpoint = 'learned_doc_appsflyers_blender/model'
    # model_checkpoint_save = 'learned_doc_appsflyers_blender/model'
    # model_checkpoint = 'learned_doc_appsflyers_blender/model'
    # model_checkpoint_token = 'learned_doc_appsflyers_blender/model'
    # model_checkpoint_save = 'learned_doc_appsflyers_blender_400m/model'
    # model_checkpoint = 'learned_doc_appsflyers_blender_400m/model'
    # model_checkpoint = 'learned_doc_appsflyers/model'
    model_checkpoint_save = 'learned_doc_appsflyers/model'
    model_checkpoint = 'learned_doc_appsflyers/model'
    # model_checkpoint = 'learned_doc_appsflyers'
    # tokenizer = AutoTokenizer.from_pretrained(
    #     'facebook/blenderbot-400M-distill')
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_token)
    # tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    # train(model=model, tokenizer=tokenizer, save_path=model_checkpoint_save)
    # chatbot = pipeline("text-generation", model=model_checkpoint)
    # prompt_conversation(generated_text, chatbot)
    prompt(question, model, tokenizer)
