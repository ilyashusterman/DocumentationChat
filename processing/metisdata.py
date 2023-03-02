from tqdm import tqdm

from models.masker import masker_result, masker_result_raw
from storage.json import load_json, save_json


def load_data():
    data = load_json('data/metis_raw.json')
    pages = []

    for page in tqdm(data['results'][0]['hits']):
        page_hits = [hit for hit in data['results'][1]['hits'] if hit['page'] == page['page']]
        raw_text = f'{page["title"]} '
        for page_hit in page_hits:
            page_body = page_hit["body"].replace('On this page', '')
            if page_hit["title"] == "":
                raw_text = f'{raw_text} {page_body}'
                continue
            try:
                page_section_text = masker_result_raw(page_hit["title"], page_body)
            except:
                page_section_text = f'{page_hit["title"]}\n{page_body}'
            raw_text = f'{raw_text} {page_section_text}'
        pages.append({
            'text': raw_text
        })
    return pages

if __name__ == '__main__':
    data = load_data()
    save_json(data, 'data/processed_metis_data.json')
