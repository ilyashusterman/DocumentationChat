import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

# Define the context


if __name__ == '__main__':

    # Tokenize the context
    input_ids = torch.tensor(tokenizer.encode(context)).unsqueeze(0)  # Batch size 1

    # Get the attention scores for each token in the context
    outputs = model(input_ids)
    attention_scores = outputs[0].detach().numpy()[0]

    # Generate questions by selecting the tokens with the highest attention scores
    generated_questions = []
    for i in range(len(attention_scores)):
        if attention_scores[i] >= 0.8:
            question = tokenizer.decode(input_ids[0][i:i+10].tolist())
            generated_questions.append(question)

    print("Generated Questions:")
    for question in generated_questions:
        print(question)
