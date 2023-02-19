import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

from storage import get_data

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

# Load the model
model = DistilBertForQuestionAnswering.from_pretrained(
    'distilbert-base-cased')
model.eval()


def answer_question(question, context):
    input_ids = tokenizer.encode(question, context)
    tensor_input_ids = torch.tensor([input_ids]).to('cuda')
    with torch.no_grad():
        outputs = model(tensor_input_ids)
        start_index = outputs[0].argmax()
        end_index = outputs[1].argmax()
        answer = tokenizer.decode(input_ids[start_index:end_index + 1],
                                  skip_special_tokens=True)
    return answer


def train_model(data):
    # Preprocess the data to get the questions and answers
    questions = []
    answers = []
    for item in data:
        questions.append(item['title'])
        answers.append(item['text'])

    # Tokenize the questions and answers
    input_ids = [tokenizer.encode(question, answer) for question, answer in
                 zip(questions, answers)]
    tensor_input_ids = torch.tensor(input_ids).to('cuda')
    with torch.no_grad():
        outputs = model(tensor_input_ids)

    # Save the model to disk
    torch.save(model.state_dict(), 'model.pt')


def load_model():
    # Load the model from disk
    model.load_state_dict(torch.load('model.pt'))
    model.eval()


if __name__ == '__main__':
    data =get_data()
    # Train the model on the data
    # train_model(data)

    # Load the model
    # load_model()

    # Ask a question and get the answer
    question = "What is the process for turning off view-through attribution in AppsFlyer?"
    answer = answer_question(question, " ".join([item['text'] for item in data]))
    print("Question:", question)
    print("Answer:", answer)
