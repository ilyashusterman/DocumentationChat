from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

def train_model(data):
    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
    for item in data:
        context = item['context']
        for qa in item['questions_and_answers']:
            question = qa['question']
            answer = qa['answer']
            qa_pipeline({'context': context, 'question': question, 'answer': answer})

    model.save_pretrained('./trained_model.pt')

def get_answer(prompt):
    model_name = "trained_model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
    context = "Please provide context for the prompt question."
    answer = qa_pipeline({'context': context, 'question': prompt})
    return answer['answer']