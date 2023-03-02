import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence
import json
import pandas as pd
from tqdm import tqdm


# Opening JSON file
with open('/home/schuldi/Downloads/process_q_a_2.json') as json_file:
    data = json.load(json_file)
    questions = pd.DataFrame(data)['question'].unique().tolist()
    print(1)

questions = questions[:200]


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModel.from_pretrained("microsoft/deberta-base")




#
# # Define the questions you want to encode
# questions = [
#     "What is the meaning of life?",
#     "What is the capital of France?",
#     "What is the Pythagorean theorem?",
#     # add your 1000 questions here
# ]

max_length = 64

# Encode the questions using the model
encoded_questions = []
for question in questions:
    input_ids = tokenizer.encode(question, return_tensors='pt')
    with torch.no_grad():
        encoded_question = model(input_ids)[0][0]  # take the first token's embeddings
    encoded_questions.append(encoded_question)

# Concatenate the encoded questions along the first axis
encoded_questions = torch.cat(encoded_questions, dim=0)

# Define the prompt question
prompt_question = "how long to test my app?"

# Encode the prompt question using the model and reshape it
with torch.no_grad():
    input_ids = tokenizer.encode(prompt_question, return_tensors='pt')
    encoded_prompt_question = model(input_ids)[0][0]
    encoded_prompt_question = encoded_prompt_question.unsqueeze(0).expand(encoded_questions.size(0), -1)

# Find the closest questions to the prompt question
similarities = torch.nn.functional.cosine_similarity(encoded_prompt_question, encoded_questions)
top_similarities, top_indices = torch.topk(similarities, k=10)

# Print the top similar questions
for similarity, index in zip(top_similarities, top_indices):
    print(f"Question: {questions[index]}")
    print(f"Similarity: {similarity.item()}")