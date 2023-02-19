import torch
from transformers import AutoTokenizer, AutoModel

# Define the data
data = [
  {
    "title": "View-through attribution guide",
    "text": "In AppsFlyer, go to Integration > Integrated Partners. Select the integrated partner. In the Integration tab, turn off Install view-through attribution. Click Save integration. In regard to SRNs: Turning off view-through attribution in AppsFlyer doesn't do the same on the network side. Make sure that you align the view-through settings in both AppsFlyer and the networks to avoid discrepancies."
  },
  {
    "title": "Getting started\u2014Onboarding for marketers",
    "text": "At a glance: Get everything set up to make the most of your AppsFlyer account. Understand the steps, timelines, and dependencies for getting your apps and your team on board. Introduction This document helps marketer (app owner) product managers plan and execute their onboarding with AppsFlyer. For every item, the owner and approximate implementation time are provided. Note that:"
  },
  {
    "title": "Using AppsFlyer with TWA",
    "text": "Measuring installs for PWA As with any other Android app, you can integrate the AppsFlyer SDK with the app to measure installs and get conversion data. To measure installs, all you need to do is integrate the SDK following the instructions here."
  },
  {
    "title": "Twitter paid vs. organic installs",
    "text": "In the AppsFlyer dashboard or reports, the interaction with the ad or tweet should be within the AppsFlyer attribution window for Twitter: 14 days click-through, 1-day view-through attribution window ...."
  }
]

# Tokenize the data
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Encode the data
encoded_data = [tokenizer.encode(item["title"] + " " + item["text"], add_special_tokens=True, max_length=512, return_tensors="pt") for item in data]

# Convert the encoded data to a tensor
inputs = torch.stack(encoded_data).squeeze(1)

# Save the model
torch.save(inputs, "data.pt")

# Load the model
inputs = torch.load("data.pt")

# Create the model
model = AutoModel.from_pretrained("bert-base-cased")

# Pass the data through the model
outputs = model(inputs)

# Get the labels
labels = [item["title"] for item in data]
