from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Conversation, ConversationalPipeline

class QA_Model:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.pipeline = ConversationalPipeline(model=self.model, tokenizer=self.tokenizer)

    def train(self, train_data):
        for data in train_data:
            conv = Conversation(data['context'])
            conv.add_user_input(data['question'])
            conv.add_user_input(data['answer'])
            self.pipeline([conv])

    def save_model(self, model_path):
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    def load_model(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.pipeline = ConversationalPipeline(model=self.model, tokenizer=self.tokenizer)

    def answer_question(self, question):
        conv = Conversation('')
        conv.add_user_input(question)
        response = self.pipeline([conv])
        return response.generated_responses[-1]


if __name__ == '__main__':
    # Create a new instance of the QA_Model class
    qa_model = QA_Model('distilbert-base-cased-distilled-squad')

    # Train the model on a conversational dataset
    train_data = [{
                      'context': 'Hugging Face is a company that provides state-of-the-art NLP technologies.',
                      'question': 'What is Hugging Face?',
                      'answer': 'Hugging Face is a company that provides state-of-the-art NLP technologies.'},
                  {'context': 'Hugging Face is based in New York City.',
                   'question': 'Where is Hugging Face based?',
                   'answer': 'Hugging Face is based in New York City.'},
                  {
                      'context': 'The Transformers library is a powerful tool for NLP tasks.',
                      'question': 'What is the Transformers library?',
                      'answer': 'The Transformers library is a powerful tool for NLP tasks.'},
                  {
                      'context': 'Fine-tuning is the process of training a pre-trained model on a specific task.',
                      'question': 'What is fine-tuning?',
                      'answer': 'Fine-tuning is the process of training a pre-trained model on a specific task.'}]
    qa_model.train(train_data)

    # Save the trained model
    qa_model.save_model('./ilya/qa_model')

    # Load the trained model
    qa_model_loaded = QA_Model('distilbert-base-cased-distilled-squad')
    qa_model_loaded.load_model('./ilya/qa_model')

    # Use the model to answer a question
    question = 'What does Hugging Face do?'
    answer = qa_model_loaded.answer_question(question)
    print(answer)
