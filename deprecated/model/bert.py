import json

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BertTokenizer, \
    get_scheduler
from transformers.pipelines.question_answering import Dataset
from torch.utils.data import DataLoader
from storage import get_data
from transformers import TrainingArguments
from torch.optim import AdamW


def train_save_model(list_of_objects, model_type='bert-base-uncased', model_name='qa_model.pt', num_epochs=5):
    remote_dataset = load_dataset("json", data_files="../processed.json", split="train")
    tokenizer = BertTokenizer.from_pretrained(model_type)
    # tokenizer.set_format("torch")
    def tokenize_dataset(data):
        # Keys of the returned dictionary will be added to the dataset as columns
        return tokenizer(data["text"].replace('\n',''), return_tensors='pt', padding=True, truncation=True)


    dataset = remote_dataset.map(tokenize_dataset)
    dataset = dataset.rename_column("title", "labels")
    dataset = dataset.remove_columns(["text", "labels"])
    dataset.set_format("torch")

    train_dataloader = DataLoader(dataset, shuffle=True,
                                  batch_size=500)
    num_epochs = 3
    model = AutoModel.from_pretrained(model_type)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader.dataset:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    # save the trained model
    torch.save(model.state_dict(), model_name)


def load_predict_model(prompt, model_type='bert-base-uncased', model_name='qa_model.pt'):
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    # convert the prompt to a tensor
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    attention_masks = torch.where(input_ids != tokenizer.pad_token_id, torch.tensor(1), torch.tensor(0))

    # load the trained model
    model = AutoModel.from_pretrained(model_type)
    model.load_state_dict(torch.load(model_name))

    # put the model in evaluation mode
    model.eval()

    # get the logits from the model
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_masks)

    # get the answer from the logits
    answer = torch.argmax(logits[0])

    return tokenizer.decode(answer)


if __name__ == '__main__':
    data = [
        {
            'questions_and_answers':[{'answer': 'At a glance: A user who views an ad (impression) and subsequently installs the app, is attributed by view-through attribution',
  'question': 'How is the user attributed'},
 {'answer': "View-through attribution measures installs, re-attributions, and re-engagements of users who viewed an ad but didn't click on it",
  'question': 'What does view-through attribution measure'},
 {'answer': '\xa0View-through attribution flowTerminologyImpression: The display of an ad to a user regardless of any click or other user engagement',
  'question': 'What is the definition of impression'},
 {'answer': '\xa0Install: In this article, means installs, re-attributions, and re-engagements',
  'question': 'What does view-through attribution measure'},
 {'answer': 'View-through attribution: Attributing installs to a media source after the user sees an impression',
  'question': 'What is the term for attribution after a user sees an ad'},
 {'answer': '\xa0View-through attribution principlesBoth device ID matching and probabilistic modeling (which you must turn on) are supported',
  'question': 'What are the two view-through attribution principles'},
 {'answer': 'Supported by partners listed in the Integrated Partners view-through attribution partner list',
  'question': 'What partners support view-through attribution'},
 {'answer': 'Impressions are candidates for view-through attribution using the following principles:The impression occurred within the view-through lookback window (default 24 hours)',
  'question': 'What are impressions candidates for view-through attribution'},
 {'answer': 'If multiple impressions are found, the most recent impression is considered',
  'question': 'What is the most recent impression considered'},
 {'answer': 'Clicks have priority over impressions',
  'question': 'What is the priority of clicks in view-through attribution'},
 {'answer': ' This means that if there are clicks in the click-through lookback window, the most recent click wins',
  'question': 'What is the result of a click on an ad'},
 {'answer': ' The most recent impression is attributed as a contributor (assist)',
  'question': 'What is the most recent impression attributed as'},
 {'answer': '\xa0View-through lookback windowFor non-SRNs, the view-through attribution lookback window duration is 1-24 hours',
  'question': 'What is the default view-through lookback window for non-SRNs'},
 {'answer': ' For SRNs, the lookback window can be longer',
  'question': 'What is the default duration of the view-through lookback window for SRNs'},
 {'answer': 'To set the window:Non-SRNs: Set the duration using impression attribution links by setting the\xa0 af_viewthrough_lookback parameter',
  'question': 'What is the default value for the view-through lookback window'},
 {'answer': " If the parameter isn't set, a default 24 hours window is used",
  'question': 'What is the default for the view-through lookback window'},
 {'answer': '\xa0SRNs: The window is determined by the SRN',
  'question': 'What is the window determined by'},
 {'answer': " We recommend setting the same value as the SRN's in AppsFlyer",
  'question': 'What is the value of the SRN'},
 {'answer': '\xa0View-through attribution reporting and metricsAggregate data: Ad networks report aggregate impression data to AppsFlyer via API, or on a per-impression basis',
  'question': 'What does the ad network report on a per impression basis'},
 {'answer': ' Consider that impression counting and reporting is defined by each partner individually',
  'question': 'How is impression counting and reporting defined'},
 {'answer': ' See definitions of impressions and clicks by major integrated partners',
  'question': 'What are the definitions of impressions and clicks'},
 {'answer': '\xa0The related metrics are available in the Overview dashboard and via other reporting tools',
  'question': 'Where can you find related metrics to impressions and clicks'},
 {'answer': ' The metrics available are:\xa0Impressions: Number of impressions displayed during a given period',
  'question': 'What are the metrics available in the Overview dashboard'},
 {'answer': 'View-throughs: The number of installs attributed using view-through attribution',
  'question': 'What is the number of installs attributed using view-through attribution'},
 {'answer': 'Raw-data:User-level data (raw data) of users attributed via view-through attribution is available and similar to that of users attributed by click',
  'question': 'What is raw data'},
 {'answer': ' However, some partners restrict the availability of this data',
  'question': 'What is the limitation of raw data'},
 {'answer': '\xa0Impression data (similar to click data) is available',
  'question': 'What is the difference between raw data and click data'},
 {'answer': 'The attributed_touch_type field contains the value impression',
  'question': 'What does the attributed_touch_type field contain'},
 {'answer': 'View-through operationsSet impression linksAd networks send impression links to AppsFlyer populated with campaign and other details similar to those included in click-through attribution links',
  'question': 'What is the default setting for view-through attribution'},
 {'answer': 'The links are sent in real-time so that if an install follows an impression, it is attributed correctly',
  'question': 'How are impression links attributed'},
 {'answer': '\xa0Consider the following in preparing links:\xa0The base URL is http://impression',
  'question': 'What is the base URL for impression.appsflyer.com'},
 {'answer': 'appsflyer',
  'question': 'What is the name of the integrated partner'},
 {'answer': 'com', 'question': 'What is the base URL of the app'},
 {'answer': ' Both HTTP and HTTPS protocols are supported',
  'question': 'What are the two protocols that are supported'},
 {'answer': 'Use a GET request',
  'question': 'How do you set the lookback window duration'},
 {'answer': 'Include device IDs where available: IDFA, GAID, IDFV, and so on',
  'question': 'What are the device IDs that are available'},
 {'answer': '\xa0Include probabilistic modeling parameters',
  'question': 'What is one way to set the lookback window duration'},
 {'answer': 'The device ID values can be hashed for better security using SHA1',
  'question': 'What is a security feature of the device IDs'},
 {'answer': ' The parameter name should begin with "sha1_" and be followed by the parameter name and the hashed value, for example, sha1_idfa, sha1_advertising_id, sha1_android_id, and sha1_imei',
  'question': 'What is the default value for the view-through attribution'},
 {'answer': '\xa0Examplehttp://impression',
  'question': 'What is the base URL for the impression attribution'},
 {'answer': 'appsflyer',
  'question': 'What is the name of the integrated partner'}],
            'context': """At a glance: A user who views an ad (impression) and subsequently installs the app, is attributed by view-through attribution.

View-through attribution measures installs, re-attributions, and re-engagements of users who viewed an ad but didn't click on it. 
View-through attribution flow

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

View-through lookback window

For non-SRNs, the view-through attribution lookback window duration is 1-24 hours. For SRNs, the lookback window can be longer.
To set the window:


Non-SRNs: Set the duration using impression attribution links by setting the  af_viewthrough_lookback parameter. If the parameter isn't set, a default 24 hours window is used. 

SRNs: The window is determined by the SRN. We recommend setting the same value as the SRN's in AppsFlyer. 



View-through attribution reporting and metrics


Aggregate data: Ad networks report aggregate impression data to AppsFlyer via API, or on a per-impression basis. Consider that impression counting and reporting is defined by each partner individually. See definitions of impressions and clicks by major integrated partners. The related metrics are available in the Overview dashboard and via other reporting tools. The metrics available are: 

Impressions: Number of impressions displayed during a given period.
View-throughs: The number of installs attributed using view-through attribution.



Raw-data:

User-level data (raw data) of users attributed via view-through attribution is available and similar to that of users attributed by click. However, some partners restrict the availability of this data. 
Impression data (similar to click data) is available.
The attributed_touch_type field contains the value impression.




View-through operations


Set impression links

Ad networks send impression links to AppsFlyer populated with campaign and other details similar to those included in click-through attribution links.
The links are sent in real-time so that if an install follows an impression, it is attributed correctly. 
Consider the following in preparing links: 

The base URL is http://impression.appsflyer.com. Both HTTP and HTTPS protocols are supported.
Use a GET request.
Include device IDs where available: IDFA, GAID, IDFV, and so on. 
Include probabilistic modeling parameters.
The device ID values can be hashed for better security using SHA1. The parameter name should begin with "sha1_" and be followed by the parameter name and the hashed value, for example, sha1_idfa, sha1_advertising_id, sha1_android_id, and sha1_imei.

 Example
http://impression.appsflyer.com/{app-id}?c={campaign name}&pid={media source}
&af_viewthrough_lookback=1d&af_prt={agency_name}&af_siteid={site id value}
&af_sub1={free value}&idfa={idfa value}





Set custom impression links

Advertisers utilize attribution impression links with their own custom media sources, for example, website banners.
When advertisers create custom attribution links, they get full impression links, which can be fired every time a user engages with their owned ads.
Custom impression links (created via OneLink) can also be used to perform view-through attribution of mobile website visitors. Learn more about creating custom impression URLs.



Turn off view-through-attribution

By default, view-through attribution is turned on. If AppsFlyer receives a valid impression attribution link, the impression participates in the attribution flow.
Since view-through attribution is enabled by default, it's important that advertisers contractually agree with the network in advance whether or not their view-through attribution should be enabled.
View-through attribution can be disabled per media source. However, if impressions are received, and view-through attribution is disabled:

Impression metrics are available in the dashboard.
The attribution process ignores these impressions.

To turn off view-through-attribution:

In AppsFlyer, go to Integration > Integrated Partners. 
Select the integrated partner.
In the Integration tab, turn off Install view-through attribution.
Click Save integration.

In regard to SRNs: Turning off view-through attribution in AppsFlyer doesn't do the same on the network side. Make sure that you align the view-through settings in both AppsFlyer and the networks to avoid discrepancies.



View VTA partner list

The list of ad network partners supporting view-through attribution is available via the dashboard. 
To view the list:


In AppsFlyer, go to Configuration > Integrated Partners. 


Select All integrations. 


In the Partner capability filter, select View-through. The list of partners displays."""
        }
    ]
    train_save_model(data)
    print(1)
