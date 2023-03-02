import json
import os

from bs4 import BeautifulSoup
from datasets import load_dataset

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
        page['body_raw'], page['body'] = page['body'], clean_html(page['body']).strip()
        data.append({
            'title': page['title'],
            'text': page['body'].strip()
        })
    return data


def get_apps_flyers_docs():
    data = load_json('data/processed.json')
    return data
def get_metics_docs():
    data = load_json('data/processed_metis_data.json')
    return data

def load_q_a():
    data = load_json('data/process_q_a_2.json')
    return data

def load_dataset_file(data_path):
    return load_dataset("json", data_files=os.path.join(CURRENT_DIR, 'data', data_path), split='train')