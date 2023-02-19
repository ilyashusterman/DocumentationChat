import torch
import transformers as trf



def generate_questions(text, model, tokenizer, max_length=64, top_k=5, top_p=0.95, num_return_sequences=1):
    input_ids = tokenizer.encode(text, return_tensors='pt')


    generated_questions = []
    with torch.no_grad():
        for i in range(num_return_sequences):
            for q_prompt in ['What is', 'How', 'for']:
                question_prompt = tokenizer.encode(q_prompt, return_tensors='pt')
                output = model.generate(input_ids=input_ids,
                                        max_length=max_length,
                                        top_k=top_k,
                                        top_p=top_p,
                                        bos_token_id=tokenizer.bos_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.pad_token_id,
                                        repetition_penalty=1.0,
                                        length_penalty=1.0,
                                        num_beams=1,
                                        early_stopping=False,
                                        decoder_start_token_id=question_prompt[0][0],
                                        use_cache=False)
                generated_questions.append(tokenizer.decode(output[0], skip_special_tokens=True))

    return generated_questions

if __name__ == '__main__':
    model_name = "gpt2"
    tokenizer = trf.GPT2Tokenizer.from_pretrained(model_name)
    model = trf.GPT2LMHeadModel.from_pretrained(model_name)
    questions = generate_questions(context, model, tokenizer)
