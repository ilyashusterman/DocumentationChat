import json
import re

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline

from processing.regex import get_question_answers_regex, clean_text
from storage.json import get_apps_flyers_docs

tokenizer = AutoTokenizer.from_pretrained(
    "mrm8488/t5-base-finetuned-question-generation-ap")
model = AutoModelWithLMHead.from_pretrained(
    "mrm8488/t5-base-finetuned-question-generation-ap")

# summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def get_question(answer, context, max_length=128):
    input_text = "answer: %s  context: %s </s>" % (answer, context)
    features = tokenizer([input_text], return_tensors='pt')

    output = model.generate(input_ids=features['input_ids'],
                            attention_mask=features['attention_mask'],
                            max_length=max_length)
    question = tokenizer.decode(output[0])
    clean_question = question.split('question: ')[-1].split('?')[0]
    return clean_question


def generate_answer(sentence):
    if len(sentence) < 200:
        return sentence
    result = summarizer(sentence)
    if not result:
        return sentence

    answer = result[0]['summary_text']
    return answer


def get_another_qa(paragraph):
    sentences = paragraph.split('..')
    q_a = []
    for sentence in sentences:
        if len(sentence) < 200:
            continue

        question = get_question(sentence, paragraph)
        q_a.append({
            'answer': sentence,
            'question': question
        })
    return q_a


def get_answers(paragraphs):
    cleaned = []
    for paragraph_raw in paragraphs:
        clean_paragraph = clean_text(paragraph_raw)
        if clean_paragraph and clean_paragraph != '':
            cleaned.append(clean_paragraph)
    return cleaned


def generate_questions(paragraphs):
    questions_answers_list = []
    for paragraph_raw in tqdm(paragraphs, 'Sentences'):
        try:
            if not paragraph_raw or len(paragraph_raw) < 10:
                continue
            paragraph = clean_text(paragraph_raw)
            regex_questions = get_question_answers_regex(data['text'])
            questions_answers_list += regex_questions
            answer = generate_answer(paragraph)
            question = get_question(answer, paragraph)
            questions_answers = {
                'answer': answer,
                'question': question
            }
            questions_answers_list.append(questions_answers)
            another_q_a = get_another_qa(paragraph)
            questions_answers_list = questions_answers_list + another_q_a
        except Exception as e:
            print('Error')
            print(e)

    return questions_answers_list


def split_paragraphs(data):
    return data['text'].split('\n\n\n')

def text_cleaner(text):
    # lower case text
    newString = text.lower()
    newString = clean_text(newString)
    newString = re.sub(r"'s\b","",newString)
    # remove punctuations
    newString = re.sub("[^a-zA-Z]", " ", newString)
    long_words=[]
    # remove short word
    for i in newString.split():
        if len(i)>=3:
            long_words.append(i)
    return (" ".join(long_words)).strip()

def create_seq(text):
    length = 30
    sequences = list()
    for i in range(length, len(text)):
        # select sequence of tokens
        seq = text[i-length:i+1]
        # store
        sequences.append(seq)
    return sequences

# create sequences

def generate_answers(data, filename):
    for part in tqdm(data, 'Page'):
        # paragraphs = split_paragraphs(part)
        # answers = get_answers(paragraphs)
        # part['answers'] = answers
        # part['paragraphs'] = paragraphs
        part['text'] = text_cleaner(part['text'])
        del part['title']
        # part['sequences'] = create_seq(part['text_clean'])

    save_file(data, filename)


def save_file(data_object, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(data_object, indent=2))


if __name__ == '__main__':
    data = get_apps_flyers_docs()
    generate_answers(data, './apps_flyers_train.json')
