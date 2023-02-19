import torch
import torch.nn as nn


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