import json
import re

from tqdm import tqdm

from storage import get_data

def get_question_answers_regex(text):
    questions = re.findall(r'\nQuestion:.*\?\n', text)
    answers = re.findall(r'\nAnswer:.*\n\n', text)
    questions_answers = []
    if not answers or not questions:
        return questions_answers

    for q, a in zip(questions, answers):
        question = q.replace('\nQuestion:', '')
        answer = a.replace('\nAnswer:', '')
        questions_answers.append({
            'question': question,
            'answer': answer
        })
    return questions_answers


if __name__ == '__main__':
    data = get_data()
    q_a = {}
    for part in tqdm(data, 'Page'):
        questions_answers = get_question_answers_regex(part['text'])
        part['questions_answers'] = questions_answers
        for q in questions_answers:
            q_a[q['question']] = q['answer']

    with open('./process_q_a_2_regex.json', 'w') as f:
        f.write(json.dumps(q_a, indent=2))
    print(1)
