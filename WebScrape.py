import transformers, torch, pandas, selenium, matplotlib
import pandas as pd

# Load the training data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Assuming your training data is called 'finance_train.csv' and has columns 'tweet' and 'sentiment'
train_data_path = "path/to/finance_train.csv"  # Replace with your path
train_df = load_data(train_data_path)

print("Loaded training data:")
print(train_df.head())

import re

# Preprocess tweets: remove URLs, mentions, special characters
def preprocess_tweet(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)  # Remove URLs
    tweet = re.sub(r'\@\w+|\#', '', tweet)  # Remove mentions and hashtags
    tweet = re.sub(r'[^A-Za-z0-9 ]+', '', tweet)  # Remove special characters
    tweet = tweet.lower()  # Convert to lowercase
    return tweet

train_df['cleaned_tweet'] = train_df['tweet'].apply(preprocess_tweet)

print("Preprocessed tweets:")
print(train_df[['tweet', 'cleaned_tweet']].head())

from transformers import BertTokenizer

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize the cleaned tweets
def tokenize_tweets(tweets, max_len=64):
    input_ids = []
    for tweet in tweets:
        encoded_tweet = tokenizer.encode(
            tweet,
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=max_len,       # Set the max length of the sequence
            padding='max_length',     # Correct replacement for pad_to_max_length
            truncation=True           # Truncate to max length if too long
        )
        input_ids.append(encoded_tweet)
    return input_ids

# Apply tokenizer to all cleaned tweets
input_ids = tokenize_tweets(train_df['cleaned_tweet'].tolist())

print("Tokenized tweets:")
print(input_ids[:5])

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# Convert input_ids to tensor
input_ids = torch.tensor(input_ids)

# Convert sentiment labels to tensor
labels = torch.tensor(train_df['sentiment'].values)

# Define batch size
batch_size = 16

# Create DataLoader for training
train_data = TensorDataset(input_ids, labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

print("DataLoader prepared with batch size:", batch_size)

from transformers import BertForSequenceClassification, AdamW

# Load BERT model for sequence classification (3 labels: negative, neutral, positive)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3,  # Number of labels (negative, neutral, positive)
    output_attentions=False,
    output_hidden_states=False
)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model loaded and moved to device:", device)

from transformers import get_linear_schedule_with_warmup

# Set up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Number of training epochs (feel free to adjust)
epochs = 3

# Total number of training steps
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

import time
import numpy as np

# Function to calculate the accuracy of predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Training loop
for epoch_i in range(epochs):
    print(f"Epoch {epoch_i + 1}/{epochs}")

    # Set model to training mode
    model.train()

    # Tracking variables
    total_loss = 0

    # Training loop over batches
    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_labels = b_labels.to(device)

        # Clear previously calculated gradients
        model.zero_grad()

        # Forward pass
        outputs = model(b_input_ids, labels=b_labels)
        loss = outputs.loss
        logits = outputs.logits

        # Accumulate the training loss
        total_loss += loss.item()

        # Backward pass to calculate gradients
        loss.backward()

        # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the optimizer
        optimizer.step()

        # Update learning rate
        scheduler.step()

    # Calculate average loss over the epoch
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss:.2f}")

print("Training complete!")

import os

# Create directory to save the model
output_dir = "./saved_model/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
import time

# Set up Selenium WebDriver
chrome_driver_path = "path/to/chromedriver"  # Replace with your ChromeDriver path
service = Service(chrome_driver_path)
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run Chrome in headless mode
driver = webdriver.Chrome(service=service, options=options)

print("WebDriver set up successfully.")

def scrape_tweets(query, num_tweets=20):
    # Open Twitter search URL
    twitter_search_url = f"https://twitter.com/search?q={query}&src=typed_query&f=live"
    driver.get(twitter_search_url)
    time.sleep(3)  # Allow the page to load

    tweets = set()  # Use a set to avoid duplicates

    while len(tweets) < num_tweets:
        # Scroll down to load more tweets
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(3)  # Wait for more tweets to load

        # Find all tweet elements on the page
        tweet_elements = driver.find_elements(By.XPATH, "//div[@data-testid='tweetText']")

        for element in tweet_elements:
            tweets.add(element.text)
            if len(tweets) >= num_tweets:
                break

    return list(tweets)

# Example: Scrape tweets with hashtag #Tesla
scraped_tweets = scrape_tweets("#Tesla", num_tweets=50)

print(f"Scraped {len(scraped_tweets)} tweets:")
for tweet in scraped_tweets[:5]:
    print(tweet)

# Preprocess scraped tweets
cleaned_tweets = [preprocess_tweet(tweet) for tweet in scraped_tweets]

print("Preprocessed tweets:")
for tweet in cleaned_tweets[:5]:
    print(tweet)

from transformers import BertTokenizer, BertForSequenceClassification

# Load the fine-tuned model and tokenizer
model_path = "./saved_model/"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Move the model to the appropriate device (GPU or CPU)
model.to(device)
print("Model and tokenizer loaded successfully.")

# Tokenize the preprocessed tweets
def tokenize_for_prediction(tweets, max_len=64):
    input_ids = []
    attention_masks = []

    for tweet in tweets:
        encoded_dict = tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',  # Correct replacement for pad_to_max_length
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert lists to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

# Apply the tokenizer to the preprocessed tweets
input_ids, attention_masks = tokenize_for_prediction(cleaned_tweets)

print("Tokenized tweets for prediction.")

from torch.utils.data import DataLoader, SequentialSampler

# Create DataLoader for the tweets
batch_size = 16
prediction_data = TensorDataset(input_ids, attention_masks)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Put the model in evaluation mode
model.eval()

# Store predictions
predictions = []

# Predict sentiment for each batch
for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask = batch

    # Don't compute gradients (no training, just inference)
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    # Get the logits (raw predictions before applying activation function)
    logits = outputs.logits

    # Move logits to CPU and convert to NumPy array
    logits = logits.detach().cpu().numpy()

    # Get the predicted class for each tweet (0 = negative, 1 = neutral, 2 = positive)
    preds = np.argmax(logits, axis=1)
    predictions.extend(preds)

print("Predictions complete.")

# Aggregate the predictions
sentiment_counts = {
    "negative": predictions.count(0),
    "neutral": predictions.count(1),
    "positive": predictions.count(2)
}

total_predictions = len(predictions)

# Calculate percentages
percent_negative = (sentiment_counts["negative"] / total_predictions) * 100
percent_neutral = (sentiment_counts["neutral"] / total_predictions) * 100
percent_positive = (sentiment_counts["positive"] / total_predictions) * 100

print(f"Sentiment analysis results for the query:")
print(f"Negative: {percent_negative:.2f}%")
print(f"Neutral: {percent_neutral:.2f}%")
print(f"Positive: {percent_positive:.2f}%")

# Decision-making function based on aggregated sentiment
def make_trading_decision(percent_negative, percent_neutral, percent_positive):
    if percent_positive > 60:
        return "Buy"
    elif percent_negative > 60:
        return "Sell"
    else:
        return "Hold"

# Use the function to make a decision based on the sentiment analysis results
trading_decision = make_trading_decision(percent_negative, percent_neutral, percent_positive)

print(f"Trading decision based on sentiment analysis: {trading_decision}")

import schedule
import time as t
import datetime as dt
import datetime

# Trade log to store all trade actions
trade_log = []

# Function to log the trades
def log_trade(decision, percent_negative, percent_neutral, percent_positive):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trade_entry = {
        "timestamp": timestamp,
        "decision": decision,
        "percent_negative": percent_negative,
        "percent_neutral": percent_neutral,
        "percent_positive": percent_positive
    }
    trade_log.append(trade_entry)
    print(f"Logged trade: {trade_entry}")

# Sentiment analysis and decision-making process
def run_sentiment_analysis():
    # Scrape tweets and preprocess them
    global trading_decision, percent_negative, percent_neutral, percent_positive
    scraped_tweets = scrape_tweets("#Tesla", num_tweets=50)
    cleaned_tweets = [preprocess_tweet(tweet) for tweet in scraped_tweets]

    # Tokenize tweets and predict sentiment
    input_ids, attention_masks = tokenize_for_prediction(cleaned_tweets)
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_dataloader = DataLoader(prediction_data, sampler=SequentialSampler(prediction_data),
                                       batch_size=batch_size)

    # Predict sentiment
    model.eval()
    predictions = []
    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        predictions.extend(preds)

    # Aggregate the predictions
    sentiment_counts = {
        "negative": predictions.count(0),
        "neutral": predictions.count(1),
        "positive": predictions.count(2)
    }
    total_predictions = len(predictions)
    percent_negative = (sentiment_counts["negative"] / total_predictions) * 100
    percent_neutral = (sentiment_counts["neutral"] / total_predictions) * 100
    percent_positive = (sentiment_counts["positive"] / total_predictions) * 100

    # Make trading decision
    trading_decision = make_trading_decision(percent_negative, percent_neutral, percent_positive)

    # Log the trading decision
    log_trade(trading_decision, percent_negative, percent_neutral, percent_positive)

# Define NYSE trading hours (9:30 AM to 4:00 PM EST)
MARKET_OPEN = dt.time(9, 30)
MARKET_CLOSE = dt.time(16, 0)

def is_market_open():
    now = dt.datetime.now().time()
    return MARKET_OPEN <= now <= MARKET_CLOSE

def run_trading_model():
    # Check if market is open
    if is_market_open():
        print(f"Market is open. Running trading model at {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        # Run sentiment analysis and make trading decisions
        run_sentiment_analysis()
        place_trade(trading_decision)
    else:
        print(f"Market is closed. No trading at {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Schedule the task every 30 minutes during market hours
schedule.every(30).minutes.do(run_trading_model)

# Run the schedule
while True:
    schedule.run_pending()
    t.sleep(60)  # Check every minute to run scheduled jobs

from alpaca_trade_api.rest import REST

API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'

# Use these keys to initialize your API client
from alpaca_trade_api.rest import REST
api = REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# Function to place simulated trades based on model output
def place_trade(decision):
    try:
        if decision == "Buy":
            # Simulated buy order
            api.submit_order(
                symbol='TSLA',
                qty=10,
                side='buy',
                type='market',
                time_in_force='gtc'  # Good 'til canceled
            )
            print(f"Simulated buy order for TSLA at {dt.datetime.now()}")
        elif decision == "Sell":
            # Simulated sell order
            api.submit_order(
                symbol='TSLA',
                qty=10,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            print(f"Simulated sell order for TSLA at {dt.datetime.now()}")
        else:
            print("Hold: No action taken.")
    except Exception as e:
        print(f"An error occurred: {e}")

import csv

def save_trade_log_to_csv(trade_log, filename="trade_log.csv"):
    if not trade_log:
        print("No trades to log.")
        return
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=trade_log[0].keys())
        writer.writeheader()
        writer.writerows(trade_log)

# Save the trade log at the end of the day
save_trade_log_to_csv(trade_log)
