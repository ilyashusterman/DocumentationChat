import json

from tqdm import tqdm

from questions.generate_questions import generate_questions, get_answers
from storage import get_data




if __name__ == '__main__':
    data = get_data()
    # all = []
    try:
        for part in tqdm(data, 'Page'):
            # questions_answers = generate_questions(part)
            answers = get_answers(part)
            part['answers'] = answers
            del part['text']
            # part['questions_answers'] = questions_answers
            # all = all + questions_answers
    except:
        print('Failed saving data...')
    with open('./process_text.json', 'w') as f:
        f.write(json.dumps(data, indent=2))
    print(1)
