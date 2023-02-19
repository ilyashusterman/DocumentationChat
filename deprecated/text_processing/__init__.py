import json

from bs4 import BeautifulSoup

from transformers import pipeline, set_seed


def load_json(filename):
    with open(filename, 'r') as f:
        return json.loads(f.read())


def clean_html(html_string):
    soup = BeautifulSoup(html_string, "html.parser")
    return soup.get_text()
