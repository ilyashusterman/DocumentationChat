import json
import os

from bs4 import BeautifulSoup

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json(file_path):
    filename = os.path.join(CURRENT_DIR, file_path)
    with open(filename, 'r') as f:
        return json.loads(f.read())


def save_json(data, file_path):
    filename = os.path.join(CURRENT_DIR, file_path)
    with open(filename, 'w') as f:
        f.write(json.dumps(data, indent=2))


def clean_html(html_string):
    soup = BeautifulSoup(html_string, "html.parser")
    return soup.get_text()


def raw_documentation(filename='./documentation_raw.json'):
    documentation_raw = load_json(filename)
    data = []
    for page in documentation_raw:
        # documentation_text = clean_html(documentation_page['body'])
        page['body_raw'], page['body'] = page['body'], clean_html(
            page['body']).strip()
        # page['snippet_raw'],  page['snippet'] = page['snippet'], preprocess_text(page['snippet'])
        # page['text_data'] = preprocess_text(page['body'])
        # page['text_data'] = preprocess_text(page['body'])
        # Combine the processed body and snippet text
        data.append({
            'title': page['title'],
            'text': page['body'].strip()
        })
    return data


def get_data():
    data = load_json('processed.json')
    return data
