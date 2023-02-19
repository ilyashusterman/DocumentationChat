import os

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel

import torch.optim as optim
import torch
import torch.nn as nn

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model_dependency = AutoModel.from_pretrained("bert-base-cased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=1):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


def initialize_model(model_originial, device):
    model = BERTClassifier(bert=model_originial)
    model = model.to(device)
    return model


def documentation_test(data, model=model_dependency):

    # Tokenize the data
    input_ids = []
    attention_masks = []
    labels = []

    for article in data:
        encoded_dict = tokenizer.encode_plus(
            article['body'] + article['snippet'],
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(article['body'])

    # Convert lists of tensors to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Split the data into train and test sets
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        input_ids, labels, test_size=0.2, random_state=42)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks,
                                                           input_ids,
                                                           test_size=0.2,
                                                           random_state=42)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    validation_data = TensorDataset(validation_inputs, validation_masks,
                                    validation_labels)
    batch_size = 32
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(model, device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    epochs = 4

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, masks, labels = data
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, masks)
            loss = criterion(outputs, labels.float().view(-1, 1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Epoch %d loss: %.4f' % (
        epoch + 1, running_loss / len(train_dataloader)))

    print('Finished Training')

    correct = 0
    total = 0

    with torch.no_grad():
        for data in validation_dataloader:
            inputs, masks, labels = data
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            outputs = model(inputs, masks)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on validation set: %d %%' % (100 * correct / total))
    return model


def predict(question, data):
    if not os.path.exists('model.pt'):
        print("Model not found, initializing and training a new model")
        model = documentation_test(data=data)
        torch.save(model.state_dict(), 'model.pt')
    else:
        print("Loading model from disk")
        model_dependency.load_state_dict(torch.load('model.pt'))
        model_dependency.eval()
        model = model_dependency

    # Tokenize the input
    input_dict = tokenizer.encode_plus(question, add_special_tokens=True,
                                        max_length=512, pad_to_max_length=True,
                                        return_attention_mask=True, return_tensors='pt')

    # Pass the input to the model
    input_ids = input_dict['input_ids'].to(device)
    attention_mask = input_dict['attention_mask'].to(device)
    with torch.no_grad():
        output = model(input_ids, attention_mask)

    # Get the answer
    answer = int(torch.sigmoid(output).item() > 0.5)

    return data[answer]['body']

if __name__ == '__main__':
    data = [{
                'body': 'At a glance: A user who views an ad (impression) and subsequently installs the app, is attributed by view-through attribution.',
                'snippet': 'At a glance: Attribution of Android devices without GAID by using OAID or IMEI. Best practice for viewing attribution',
                'title': 'View-through attribution guide',
        }]
    uestion = "What is view-through attribution?"
    print(1)
