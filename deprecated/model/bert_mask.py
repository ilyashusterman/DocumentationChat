import torch
import transformers as tfm

# Load the model
tokenizer = tfm.BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = tfm.BertForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")

# Define a function to predict the answer for a given question and context
def predict_answer(question, context, model):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0,
                            answer_start_index: answer_end_index + 1]
    answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
    loss = outputs.loss
    return answer, outputs.loss



def train(data, context, model):
    for obj in data:
        question = obj['question']
        answer = obj['answer']
        predicted_answer, loss_result = predict_answer(question, context, model)
        if answer != predicted_answer:
            model.train()
            input_ids = tokenizer.encode(question, context)
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            label = torch.tensor([answer])
            loss, logits = model(input_ids, labels=label)
            loss.backward()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.step()

    torch.save(model, './model.pt')
    # torch.load('./model.pt')

if __name__ == '__main__':
    data =[{'answer': 'At a glance: A user who views an ad (impression) and subsequently installs the app, is attributed by view-through attribution',
  'question': 'How is the user attributed'},
 {'answer': "\n\nView-through attribution measures installs, re-attributions, and re-engagements of users who viewed an ad but didn't click on it",
  'question': 'What does view-through attribution measure'},
 {'answer': '\xa0\nView-through attribution flow\n\nTerminology\n\n\nImpression: The display of an ad to a user regardless of any click or other user engagement',
  'question': 'What is the term for the display of an ad to a user'},
 {'answer': '\xa0\n\nInstall: In this article, means installs, re-attributions, and re-engagements',
  'question': 'What does view-through attribution measure'},
 {'answer': ' See definitions of impressions and clicks by major integrated partners',
  'question': 'What are the definitions of impressions and clicks'},
 {'answer': 'The related metrics are available in the Overview dashboard and via other reporting tools',
  'question': 'Where can you find related metrics to impressions and clicks'}]
    context="""At a glance: A user who views an ad (impression) and subsequently installs the app, is attributed by view-through attribution.

View-through attribution measures installs, re-attributions, and re-engagements of users who viewed an ad but didn't click on it. View-through attribution flow

Terminology


Impression: The display of an ad to a user regardless of any click or other user engagement.

Install: In this article, means installs, re-attributions, and re-engagements.

View-through attribution: Attributing installs to a media source after the user sees an impression.

View-through attribution principles

Both device ID matching and probabilistic modeling (which you must turn on) are supported.
Supported by partners listed in the Integrated Partners view-through attribution partner list.

Impressions are candidates for view-through attribution using the following principles:

The impression occurred within the view-through lookback window (default 24 hours).
If multiple impressions are found, the most recent impression is considered.
Clicks have priority over impressions. This means that if there are clicks in the click-through lookback window, the most recent click wins. The most recent impression is attributed as a contributor (assist).

View-through attribution can be disabled per media source. However, if impressions are received, and view-through attribution is disabled:

Impression metrics are available in the dashboard.
The attribution process ignores these impressions."""
    train(data=data, context=context, model=model)
    print()