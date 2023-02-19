import transformers
import torch
from torch.nn.utils.rnn import pad_sequence
from main import load_json, clean_html
from transformers import pipeline, set_seed

from storage import get_data

model = None
tokenizer = None


def answer_question(question, model_path = 'model.pt'):
    data = get_data()
    # Combine titles and texts
    corpus = [item['title'] + ' ' + item['text'] for item in data]

    # Train the model
    model = pipeline("question-answering", model="bert-base-cased", device=0)
    model.train(corpus)
    # Save the model
    torch.save(model.model, model_path)


    # Example usage
    question = 'What is view-through attribution in AppsFlyer?'
    context = 'View-through attribution guide\nboard. To view the list: In AppsFlyer, go to Configuration > Integrated Partners. Select All integrations. In the Partner capability filter, select View-through. The list of partners displays.....'

    print(get_answer(question, context))



def answer_question(question):
    best_answer = None
    try:
        model = torch.load("my_model")
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    except:
        context = get_raw_context_tokenized()
        data = []
        for d in context:
            data.append({
                'title': d['title'],
                'text': d['body'].strip()
            })
        # Train the model on the data
        answers = []
        texts = []
        for item in data:
            answers.append(item["text"])
            texts.append(item["title"] + " " + item["text"])

        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

        input_ids = []
        attention_masks = []

        for text in texts:
            encoded_dict = tokenizer.encode_plus(question, text, add_special_tokens=True, max_length=512, return_attention_mask=True, truncation=True, padding='max_length', return_tensors='pt')
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_masks)
            start_scores, end_scores = output[0], output[1]

        best_answer = None
        max_score = float("-inf")
        for i in range(input_ids.shape[0]):
            all_tokens = tokenizer.convert_ids_to_tokens(input_ids[i].tolist())
            answer_tokens = all_tokens[torch.argmax(start_scores[i]) : torch.argmax(end_scores[i])+1]
            answer = " ".join(answer_tokens)

            score = sum([start_scores[i][j].item() for j in range(torch.argmax(start_scores[i]), torch.argmax(end_scores[i])+1)])
            if score > max_score:
                best_answer = answer
                max_score = score

        torch.save(model, "my_model")

    # Use the trained model to generate an answer
    encoded_dict = tokenizer.encode_plus(question, best_answer, add_special_tokens=True, max_length=512, return_attention_mask=True, truncation=True, padding='max_length', return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_masks)
        start_scores, end_scores = output[0], output[1]

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    answer_tokens = all_tokens[torch.argmax(start_scores[0]): torch.argmax(
        end_scores[0]) + 1]
    answer = " ".join(answer_tokens)

    return answer



if __name__ == '__main__':

    question = "What is view-through attribution?"
    result = answer_question(question=question)
    print()