import torch
from transformers import AutoTokenizer, AutoModel

from storage import get_data
PATH = './model.pt'

def train_and_save_model(data):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Encode the input data as input_ids and attention masks
    input_ids = []
    attention_masks = []
    for item in data:
        encoded_dict = tokenizer.encode_plus(
            item['title'] + ": " + item['text'],
            max_length=512,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Train the model
    model = AutoModel.from_pretrained("bert-base-cased")
    model.train()

    # Save the model
    torch.save(model.state_dict(), PATH)


def load_model_and_answer_question(question):
    # Load the model
    model = AutoModel.from_pretrained("bert-base-cased")
    model.load_state_dict(torch.load(PATH))
    model.eval()

    # Tokenize the question
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer.encode(question, return_tensors='pt')

    # Use the model to generate an answer
    with torch.no_grad():
        outputs = model(input_ids)
        answer = tokenizer.decode(outputs[0].argmax(dim=1).tolist())

    return answer


if __name__ == '__main__':

    data = get_data()

    train_and_save_model(data)

    # Example usage
    question = "What is view-through attribution?"
    answer = load_model_and_answer_question(question)
    print("Answer: ", answer)
