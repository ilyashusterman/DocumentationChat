from storage import get_data
from tqdm import tqdm

import torch

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup
)


PATH = './model_doc'
PATH_TOKEN = './model_token'
def split_text(text, chunk_size=1024):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks



def train_model(contexts):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    input_ids = []
    for context in contexts:
        chunks = split_text(context['text'], chunk_size=1024)
        for chunk in chunks:
            encoded_chunk = tokenizer.encode(chunk, add_special_tokens=True)
            input_ids.append(torch.tensor(encoded_chunk))
    # Set the model to evaluation mode
    model.eval()

    # Generate the masked text
    with torch.no_grad():
        for input_id in tqdm(input_ids):
            outputs = model(input_id)

    # Calculate the loss based on the masked text
    loss, logits = outputs[:2]

    # Set the model to training mode
    model.train()

    # Train the model on the masked text
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(5):
        optimizer.zero_grad()
        for input_id in tqdm(input_ids):
            outputs = model(input_id)
            loss, logits = outputs[:2]
            # loss.backward()
            optimizer.step()

    # Save the model
    model.save_pretrained(PATH)
    tokenizer.save_pretrained(PATH_TOKEN)



def generate_text(prompt):
    model = GPT2LMHeadModel.from_pretrained(PATH)
    tokenizer = GPT2Tokenizer.from_pretrained(PATH_TOKEN)
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(input_ids, max_length=100)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output


if __name__ == '__main__':
    data = get_data()
    train_model(data[:3])
    while True:
        question = input("\nPlease enter your question: ")
        if question == "quit":
            break
        else:
            print(generate_text(question))
