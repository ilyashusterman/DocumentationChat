from transformers import AutoTokenizer, AutoModelForCausalLM, \
    TrainingArguments, Trainer, DataCollatorForLanguageModeling

from deprecated.questions.prompt import prompt
from storage.json import load_dataset_file


def train(model, tokenizer, model_checkpoint, save_path="finetuned-wikitext2"):
    lm_datasets = get_dataset(tokenizer, 'apps_flyers_train.json')
    lm_datasets_validation = get_dataset(tokenizer, 'apps_flyers_test.json')
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=False)
    model_name = model_checkpoint.split("/")[-1]
    model_name_save = f"{model_name}-{save_path}"
    training_args = TrainingArguments(
        model_name_save,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=lm_datasets,
        eval_dataset=lm_datasets_validation,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(
        model_name_save
    )
    print(f'Model name:{model_name_save}')


def get_dataset(tokenizer, data_path):
    dataset_train = load_dataset_file(data_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = dataset_train.map(tokenize_function, batched=True,
                                           num_proc=4,
                                           remove_columns=['text'])
    block_size = 128

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in
                                 examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in
                range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    return lm_datasets


def question(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs)
    # outputs = model.generate(inputs, max_new_tokens=100, do_sample=True,
    #                          top_k=50, top_p=0.95)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for r in result:
        print(r)
    return result


if __name__ == '__main__':
    save_path = "appsflyers2"
    # model_checkpoint = "distilgpt2"
    model_checkpoint = "distilgpt2-appsflyers2"
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    # train(model=model, tokenizer=tokenizer, model_checkpoint=model_checkpoint,
    #       save_path=save_path)
    prompt(question, model, tokenizer)
