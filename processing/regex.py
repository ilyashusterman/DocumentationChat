import re


def clean_text(text):
    return text.replace('\n\n', ' ').replace('\n\n\n', ' ').replace('\n', ' ').replace(':', ' ').replace('.', ' ').replace(',', ' ').replace(
        ' ', '').replace('\xa0', '').replace('    ', ' ').replace('   ', ' ').replace('  ', ' ')

def get_question_answers_regex(obj):
    questions = re.findall(r'\nQuestion:.*\?\n', obj)
    answers = re.findall(r'\nAnswer:.*\n', obj)
    questions_answers = []
    if not answers or not questions:
        return questions_answers

    for q, a in zip(questions, answers):
        questions_answers.append({
            'question': q,
            'answer': a
        })
    return questions_answers