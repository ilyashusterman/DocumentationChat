import math

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, AutoConfig, \
    GPT2LMHeadModel, AutoTokenizer
from transformers import TrainingArguments, Trainer


def train(tokenizer, model, output_dir):
    data_path = './apps_flyers.json'
    data_path_test = './apps_flyers_test.json'
    # questions_answers = load_json('./process_q_a_2.json')
    # data = get_data()
    # text_data = [{'text': clean_text(d['text'])} for d in data]
    # # text_data = text_data + [{'text': f"{a['question']} {a['answer']}"} for a
    #                           in questions_answers]
    # save_json(text_data, data_path)
    # save_json(text_data[:100], data_path_test)

    lm_dataset = load_dataset_lm(data_path, tokenizer, 'train')
    lm_dataset_test = load_dataset_lm(data_path_test, tokenizer, 'train')
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size / 1000 ** 2:.1f}M parameters")
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=f'{output_dir}/results',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=lm_dataset,
        eval_dataset=lm_dataset_test,
    )
    trainer.train()
    # trainer.save_model()
    trainer.save_model(
        output_dir
    )
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    # trainer.save_state(save_path)


def get_model(model_checkpoint, tokenizer, context_length=128):
    config = AutoConfig.from_pretrained(
        model_checkpoint,
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    return model


def load_dataset_lm(data_path, tokenizer, split, context_length=128):
    dataset = load_dataset("json", data_files=f'.{data_path}', split=split)

    def tokenize(examples):
        outputs = tokenizer(examples["text"],
                            truncation=True,
                            max_length=context_length,
                            return_overflowing_tokens=True,
                            return_length=True)
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = dataset.map(
        tokenize,
        batched=True,
        remove_columns=['text'],
    )
    print()
    return tokenized_datasets


def question(question: str, pipe):
    result_pipe = pipe(question, num_return_sequences=1)[0]["generated_text"]
    print(result_pipe)
    return result_pipe


def prompt(func, pipe, msg="\nUser: "):
    while True:
        question = input(msg)
        if question == "quit":
            break
        else:
            func(question, pipe)


if __name__ == '__main__':
    tokenizer_checkpoint = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    model_checkpoint = 'gpt2'
    # model_checkpoint = './gpt2_model_appsflyer'
    model_save_checkpoint = './gpt2_model_appsflyer'
    model = get_model(model_checkpoint=model_checkpoint,
                      tokenizer=tokenizer,
                      context_length=128)

    train(tokenizer=tokenizer,
          model=model,
          output_dir=model_save_checkpoint
    )
    # device = torch.device(
    #     "cuda") if torch.cuda.is_available() else torch.device("cpu")
    # pipe = pipeline(
    #     "text-generation", model=model_save_checkpoint,
    #     device=device
    # )
    # prompt(question, pipe)
