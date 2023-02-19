import torch
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead
from transformers import pipeline, set_seed

from transformers import GPT2Tokenizer, GPT2Model

from storage import get_data
PATH = './model.pt'
MODEL_NAME= "bert-base-uncased"

from transformers import GPT2Tokenizer, GPT2Model

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
generator = pipeline('text-generation', model='gpt2')

def generate_question(prompt):
    resposes = generator(prompt, max_length=200,
              num_return_sequences=5)
    encoded_input = tokenizer(prompt, return_tensors='pt')
    output = model(**encoded_input)
    return question




def train_and_save_model(data, model_name=MODEL_NAME, max_length=512, batch_size=16):
    # Create input text
    input_text = []
    for item in data:
        context_raw = item['text']
        context = context_raw[:4000]
        # prompt = f"Generate a questions from the following context: {context}"
        prompt = f'Generate a question from the following context from "START" to "END": """START {context} END """'
        questions = generate_question(prompt)
        print()


    # Tokenize input text
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = []
    attention_masks = []
    for text in input_text:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert input ids and attention masks to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Train model
    model = AutoModel.from_pretrained(model_name)
    model.train()

    # Save model
    torch.save(model.state_dict(), f"{model_name}_state_dict.pt")



def get_answer(output, tokenizer):
    logits = output[0]
    logits = logits[0, -1, :]
    index = torch.argmax(logits).item()
    token = tokenizer.convert_ids_to_tokens([index])[0]
    return token

def load_and_answer_question(question, model_name=MODEL_NAME, max_length=512):
    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.load_state_dict(torch.load(f"{model_name}_state_dict.pt"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize question
    input_ids = tokenizer.encode(question, return_tensors='pt')

    # Get model output
    with torch.no_grad():
        output = model(input_ids)

    # Get answer
    answer = get_answer(output, tokenizer)

    return answer



if __name__ == '__main__':

    # data = get_data()
    # model = train_and_save_model(data=data)
    # Example usage
    question = "What is view-through attribution?"
    answer = load_and_answer_question(question)

    print("Answer: ", answer)
