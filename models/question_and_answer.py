import pickle

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, \
    AutoModelForQuestionAnswering

from deprecated.questions.prompt import prompt_conversation
from models.masker import masker_result
from storage.json import get_apps_flyers_docs, get_metics_docs

unmasker = pipeline('fill-mask', model='bert-base-uncased')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
nlp = spacy.load("en_core_web_sm")


def question(prompt, qa_pipeline, get_context, max_length_answer=600,
             min_length_answer_result=100):
    question_tokens = preprocess(prompt).replace('?', ' ').replace('  ', '')
    context, similarities_contexts = get_context(prompt)
    if similarities_contexts:
        print()
    answer = qa_pipeline({'context': context, 'question': prompt},
                         max_answer_len=600)
    if (answer['end'] - answer['start']) < min_length_answer_result:
        result = context[answer['start']:answer['end'] + max_length_answer]
    else:
        result = answer['answer']

    formatted = masker_result(question_tokens, result)
    summary = summarizer(formatted, do_sample=False)
    summary_text = summary[0]['summary_text']

    if summary_text.count(question_tokens) > 0:
        summary_text = summary_text[summary_text.index(question_tokens):]
        formatted = formatted.replace(question_tokens, summary_text)
        print(formatted)
        return formatted

    print(summary_text)
    print('>>>>Docs:')
    print(formatted)
    return formatted


def preprocess(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])


def ask_question(question, docs, vectorizer, max_similarities=3):
    # Preprocess the input question
    question_processed = preprocess(question)

    # Find the most similar document to the input question
    question_vector = vectorizer.transform([question_processed])
    similarity_scores_options = cosine_similarity(question_vector, vectors)
    similarities_docs = []
    if similarity_scores_options.shape[0] > 1:
        similarities_docs = [
            docs[score.argmax()] for score in
            similarity_scores_options[1:max_similarities]
        ]
    similarity_scores = similarity_scores_options[0]
    most_similar_index = similarity_scores.argmax()
    most_similar_doc = docs[most_similar_index]

    # Extract the answer from the most similar document

    return most_similar_doc, similarities_docs


def process_documents(path, get_data_func, is_save=None):
    docs_path = f'docs/docs_{path}.pickle'
    docs_path_processed = f'docs/docs_processed_{path}.pickle'

    if is_save is None:
        with open(docs_path, 'rb') as f:
            docs = pickle.loads(f.read())
        with open(docs_path_processed, 'rb') as f:
            processed_docs = pickle.loads(f.read())

        return docs, processed_docs

    processed_docs = []
    docs = []
    data = get_data_func()
    for doc in tqdm(data, 'DOC'):
        processed_docs.append(preprocess(doc['text']))
        docs.append(doc['text'])
    with open(docs_path, 'wb') as f:
        f.write(pickle.dumps(docs))
    with open(docs_path_processed, 'wb') as f:
        f.write(pickle.dumps(processed_docs))

    return docs, processed_docs


if __name__ == '__main__':
    docs, processed_docs = process_documents(
        # path='appsflyers',
        # get_data_func=get_apps_flyers_docs,
        path='metisdata',
        get_data_func=get_metics_docs,
        # is_save=True
    )

    model_name = "distilbert-base-cased-distilled-squad"
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(processed_docs)
    similarity_matrix = cosine_similarity(vectors)


    def get_context(query):
        answer = ask_question(query, docs, vectorizer)
        return answer


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    pipe = pipeline('question-answering', model=model, tokenizer=tokenizer)

    prompt_conversation(question, pipe, get_context)
