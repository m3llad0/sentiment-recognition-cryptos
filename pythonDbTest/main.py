# This is a sample Python script.
from itertools import count

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import zstandard
import json
from datetime import datetime
#import re
import regex as re

# Assuming cleaned_text is the text you want to clean
from sklearn.tree import DecisionTreeRegressor

import spell_check
import nltk
from tqdm import tqdm
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.datasets import make_regression
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import tensorflow as tf
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import ast

FIRST_TRY = True
SECOND_TRY = False

DetectorFactory.seed = 0

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
	chunk = reader.read(chunk_size)
	bytes_read += chunk_size
	if previous_chunk is not None:
		chunk = previous_chunk + chunk
	try:
		return chunk.decode()
	except UnicodeDecodeError:
		if bytes_read > max_window_size:
			raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
#		log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
		return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)

def read_lines_zst(file_name):
	with open(file_name, 'rb') as file_handle:
		buffer = ''
		reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
		while True:
			chunk = read_and_decode(reader, 2**27, (2**29) * 2)

			if not chunk:
				break
			lines = (buffer + chunk).split("\n")

			for line in lines[:-1]:
				yield line, file_handle.tell()

			buffer = lines[-1]

		reader.close()

INPUT_OUTPUT_FILES = [
#    "Bitcoin_comments",
#    "Bitcoin_submissions",
    "ethereum_comments",
    "ethereum_submissions"
]

def write_my_file_to_csv():
    for input_output_file in INPUT_OUTPUT_FILES:
        input_path = 'subreddits23/' + input_output_file + '.zst'
        output_path = 'output/latest/' + input_output_file + '_clean_reduced_final.csv'
        start_date = datetime.strptime("2021-10-01", "%Y-%m-%d")
        final_date = datetime.strptime("2022-09-01", "%Y-%m-%d")

        desired_keys = [
            'created_utc',
            'subreddit',
            'title',
            'body'
        ]

        data_list = []

        try:
            # this is the main loop where we iterate over every single line in the zst file
            tries = 0
            written = False
            for line, file_bytes_processed in read_lines_zst(input_path):
                try:
                    # load the line into a json object
                    obj = json.loads(line)
                    # turn the created timestamp into a date object
                    created = datetime.utcfromtimestamp(int(obj['created_utc']))
                    # skip if we're before the start date defined above
                    if start_date <= created <= final_date:
                        partial_dict = {key: obj[key] for key in desired_keys if key in obj}
                        data_list.append(partial_dict)
#                       data_list.append(line)
                        if not written:
                            print(f"Initial date: {created} - {str(obj)}")
                            written = True
                    tries = tries + 1
                    if tries % 100000 == 0:
                        print(f"Tries: {tries}")
                # just in case there's corruption somewhere in the file
                except (KeyError, json.JSONDecodeError) as err:
                    print(f"WFT: {err}")
            df = pd.DataFrame(data_list)
            df.to_csv(output_path, index=False)
            print(f"Final date: {created}")
        except Exception as err:
            print(f"Err: {err}")
    # elif SECOND_TRY:
    #    input_path = 'CryptoCurrency_submissions_2023.csv'
    #    df = pd.read_csv(input_path)
    #    print(df.head())

def clean_body(text):
    if pd.isna(text):  # Check for NaN values
        return text  # Return as is if NaN
    if isinstance(text, str):  # Ensure the input is a string
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs

        cleaned_text = re.sub(r'[\n\r]+', ' ', text)  # Replace newlines with space
        cleaned_text = re.sub(r'[^\w\s#\p{So}\p{C}]', ' ', cleaned_text)
        cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)  # Remove punctuation and symbols
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Remove extra whitespace
        return cleaned_text.strip()  # Remove leading/trailing spaces
    return text  # Return original if not a string



def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    if isinstance(text, str):  # Check if the input is a string
        sentiment = analyzer.polarity_scores(text)
        return sentiment['compound']  # Return the 'compound' score
    else:
        return None  # Return None for non-string inputs

# Modified get_sentiment function
def get_sentiment_full(text):
    analyzer = SentimentIntensityAnalyzer()
    if isinstance(text, str):  # Check if the input is a string
        sentiment = analyzer.polarity_scores(text)
        # Return a pandas Series with four columns
        return pd.Series({
            'compound': sentiment['compound'],
            'neg': sentiment['neg'],
            'pos': sentiment['pos'],
            'neu': sentiment['neu']
        })
    else:
        return pd.Series({
            'compound': None,
            'neg': None,
            'pos': None,
            'neu': None
        })

# Load the sentiment analysis pipeline with the 'bertweet-base-sentiment-analysis' model
tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis", )
model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

specific_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def get_sentiment_bert(text):
    # Ensure the input is a list, if it's not a list, convert it
    if isinstance(text, str):

        # Process each text
        try:
            # Detect language of the text
            language = detect(text)
#            print(f"{text} - Language: {language}")

            if language == "en":
                # If the text is in the supported language, run the sentiment model
                result = specific_model(text)
                return result
#            else:
                # If the text is not in the supported language, skip it
#                print(f"Skipping text (unsupported language: {language}): {text}")

        except LangDetectException:
            # Handle exception in case language detection fails
            print(f"Language detection failed for text: {text}")
        except Exception as e:
            # Catch any other unexpected errors
            print(f"An error occurred: {e}")
    return None

def contains_numbers(s):
    return bool(re.search(r'\d', s))

# Function to check if the string is valid (English words, emojis, but no numbers)
def is_valid(s):
    if not isinstance(s, str):  # If s is None or NaN (not a string), it's invalid
        return False
    # Remove rows that contain any numbers
#    if contains_numbers(s):
#        return False
    # Keep strings with English letters and optional punctuation
    elif re.match(r'^[a-zA-Z\s\.,!?\'"-]*$', s):
        return True
    # Keep strings with emojis (Unicode characters from U+1F600 onwards)
    elif bool(re.search(r'[\U00010000-\U0010ffff]', s)):
        return True
    # Remove non-ASCII or other scripts (like Chinese, Arabic)
    return False

def get_analysis():
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    ALREADY_RUN_ONCE=False

    tqdm.pandas(desc="Processing")
    for COMMON_NAME in SUBREDDIT_NAME:
        if not ALREADY_RUN_ONCE:
            comments_df = pd.read_csv('output/' + COMMON_NAME + '_comments_clean_reduced_final.csv')
            submissions_df = pd.read_csv('output/' + COMMON_NAME + '_submissions_clean_reduced_final.csv')
            submissions_df['origin'] = 'submissions'
            comments_df['origin'] = 'comments'

            # Rename 'title' in submissions_df to 'body' to unify the columns
            submissions_df = submissions_df.rename(columns={'title': 'body'})

            # Keep only the columns of interest for the final output
            submissions_df = submissions_df[['created_utc', 'subreddit', 'body', 'origin']]
            comments_df = comments_df[['created_utc', 'subreddit', 'body', 'origin']]

            # Convert created_utc to int (if it can be converted)
#            combined_df = submissions_df
            combined_df = pd.concat([submissions_df, comments_df], axis=0, ignore_index=True)

            print(f'Start Number of elements in the combined DataFrame: {len(combined_df)}')

            # Handle potential mixed types in 'created_utc'
            combined_df['created_utc'] = pd.to_numeric(combined_df['created_utc'], errors='coerce')

#            combined_df['body'] = combined_df['body'].progress_apply(clean_body)
            #################################################
            ## Applying filtering from roberta ipybn file
            print(f'Before Roberta clean: {len(combined_df)}')
            combined_df.dropna(subset=['body'], inplace=True)
            combined_df = combined_df[~combined_df['body'].str.contains(r'\b(?:deleted|removed)\b', case=False, na=False)]
            combined_df = combined_df[combined_df['body'].str.len() > 5]  # Filter out comments shorter than 3 characters
            keywords = ["price", "up", "down", "increase", "decrease", "rise", "fall", "bull", "bear", "market",
                        "value", "worth"]
            robertaPattern = re.compile(r'\b(?:' + '|'.join(keywords) + r')\b', re.IGNORECASE)
            combined_df = combined_df[combined_df['body'].progress_apply(lambda x: bool(robertaPattern.search(str(x))))]
            print(f'After Roberta clean: {len(combined_df)}')
            #################################################

            # Drop rows where created_utc is NaN (if any)
            combined_df = combined_df.dropna(subset=['created_utc'])

            # Sort the combined DataFrame by 'created_utc'
            combined_df = combined_df.sort_values(by='created_utc')

            combined_df.to_csv('output/latest/H_' + COMMON_NAME + '_submissions_before_vader.csv', index=False)
        else:
            combined_df = pd.read_csv("output/latest/H_" + COMMON_NAME + "_submissions_before_vader.csv")
            print(f'A Start Number of elements in the combined DataFrame: {len(combined_df)}')
        combined_df = combined_df[combined_df['body'].progress_apply(is_valid)]
        print(f'End Number of elements in the combined DataFrame: {len(combined_df)}')
        combined_df.to_csv('output/latest/H_' + COMMON_NAME + '_submissions_before_vader_2.csv', index=False)

#        combined_df['sentiment'] = combined_df['body'].progress_apply(get_sentiment)
        combined_df[['compound', 'neg', 'pos', 'neu']] = combined_df['body'].progress_apply(get_sentiment_full)

        # Save the combined DataFrame to a new CSV file
        combined_df.to_csv('output/latest/H_' + COMMON_NAME + '_comments_vader.csv', index=False)
        print(f'End Number of elements in the combined DataFrame: {len(combined_df)}')

# Function to convert the volume with 'K' or 'M' to numeric
def convert_vol(vol):
    if 'K' in vol:
        return float(vol.replace('K', '').strip()) * 1000
    elif 'M' in vol:
        return float(vol.replace('M', '').strip()) * 1000000
    elif 'B' in vol:
        return float(vol.replace('B', '').strip()) * 1000000000
    else:
        return float(vol)

def extract_sentiment_score(sentiment_str):
    try:
        # Convert string representation of a list to an actual list (using ast.literal_eval)
        sentiment_list = ast.literal_eval(sentiment_str)
        # Extract the 'score' value from the first dictionary in the list
        return sentiment_list[0]['score'] if sentiment_list else None
    except Exception as e:
        return None

# Apply the conversion function to the 'Vol.' column
def get_graphs_and_all():
#   for COIN in SUBREDDIT_NAME:
    COIN = 'twitter'
    print(f"Loading {COIN} in csv")
    PROCESSING = False
    ROBERTA = False
    # Load data
    if PROCESSING:
        df = pd.read_csv("output/latest/H_" + COIN + "_comments_vader.csv")
        price_df = pd.read_csv("cryptoPrices/" + COIN + "_price.csv")

#       ONLY IF BERT!
#        df['sentiment'] = df['sentiment'].apply(extract_sentiment_score)

        # Filter sentiments
#        df = df[(df['sentiment'] >= 0.3) | (df['sentiment'] < -0.15)]

        # Prepare date and average sentiment
        df['date'] = pd.to_datetime(df['created_utc'], unit='s').dt.date
        df['date'] = pd.to_datetime(df['date'])

        # Set the 'date' column as the index for grouping
        df.set_index('date', inplace=True)

        # Group by week and calculate average sentiment
#        sentiment_df = df.resample('D')['pos'].mean().reset_index()
#        sentiment_df.columns = ['date', 'avg_sentiment']
        sentiment_df = df.resample('D')[['pos', 'neg', 'neu']].mean().reset_index()
        sentiment_df['avg_sentiment'] = sentiment_df['neg']*-0 + sentiment_df['pos']*1

        # Drop the 'pos' and 'neg' columns if they are no longer needed
        sentiment_df = sentiment_df[['date', 'avg_sentiment']]

        # Rename columns for clarity
        sentiment_df.columns = ['date', 'avg_sentiment']

        # Reset the index of sentiment_df to merge properly
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

        # Merge price and sentiment DataFrames
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        price_df['Price'] = price_df['Price'].str.replace(',', '').astype(float)
        price_df.set_index('Date', inplace=True)
        price_df = price_df.resample('D')['Price'].mean().reset_index()
        merged_df = pd.merge(price_df, sentiment_df, left_on='Date', right_on='date', how='outer')

        merged_df = merged_df[['date', 'Price', 'avg_sentiment']]
    elif ROBERTA:
       merged_df = pd.read_csv('sentiment_data_daily_roberta.csv')
       price_df = pd.read_csv("cryptoPrices/" + COIN + "_price.csv")
       merged_df['avg_sentiment'] = merged_df['sentiment']
       merged_df['date'] = pd.to_datetime(merged_df['date'])
       price_df['Date'] = pd.to_datetime(price_df['Date'])
       price_df['Price'] = price_df['Price'].str.replace(',', '').astype(float)
       price_df.set_index('Date', inplace=True)
       price_df = price_df.resample('D')['Price'].mean().reset_index()
       merged_df = pd.merge(price_df, merged_df, left_on='Date', right_on='date', how='outer')

    else:
        merged_df = pd.read_csv('twitter_data_gerardo_goodcorr_.csv')

    ## -7 is 7 days later
    ## 14 is 14 days before
    ## -28*2 is good for tests
    merged_df['Price_7_days_later'] = merged_df['Price'].shift(7 * 0)

    # Drop rows with missing values
    merged_df = merged_df.dropna(subset=['avg_sentiment', 'Price', 'Price_7_days_later'])
    merged_df.to_csv("output/latest/merged_dataframes_" + COIN + ".csv")

    merged_df['date'] = pd.to_datetime(merged_df['date'])

    # Calculate the difference in price
    if PROCESSING:
        merged_df['Price_Diff'] = merged_df['Price_7_days_later']#.diff()
    else:
        merged_df['Price_Diff'] = merged_df['Price_7_days_later']

    if PROCESSING:
        window_size = 100  # Example window size for rolling average - 100 before
    elif ROBERTA:
        window_size = 50
    else:
        window_size = 5

    merged_df['avg_sentiment_rolling'] = merged_df['avg_sentiment'].rolling(window=window_size).mean()
    merged_df['Price_Diff_rolling'] = merged_df['Price_Diff'].rolling(window=window_size).mean()

    # Drop the first row which will have a NaN value in Price_Diff
    merged_df = merged_df.dropna()

    # Prepare the features and target variable
    X = merged_df[['avg_sentiment_rolling']]
    y = merged_df['Price_Diff_rolling']

    # Normalize the data
    scaler = MinMaxScaler()
    Y_normalized = scaler.fit_transform(y.values.reshape(-1, 1))

    X_scaler = MinMaxScaler()
    X_normalized = X_scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y_normalized, test_size=0.15, random_state=20)

    params_dt = {
        'criterion': ['squared_error'],  # Use 'squared_error' for regression
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    # Define the KNN model and hyperparameters
    params_knn = {
    'n_neighbors': [1, 3, 5, 7, 9],  # Number of neighbors
    'weights': ['uniform', 'distance'],  # Weight function
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
    }
    params_rf = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_features': ['auto', 'sqrt'],  # Number of features to consider at every split
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]     # Minimum number of samples required to be at a leaf node
    }
    rf_regressor = RandomForestRegressor(random_state=42)
    knn_regressor = KNeighborsRegressor()
    dt_regressor = DecisionTreeRegressor(random_state=42)

    # Create a dictionary of the models and their respective parameters
    models = {
#        'Random Forest': {'model': rf_regressor, 'params': params_rf},
        'KNN': {'model': knn_regressor, 'params': params_knn},
        'Decision Tree': {'model': dt_regressor, 'params': params_dt}
    }

    # Initialize variables to track the best model and score
    best_model = None
    best_r2 = 0

    # Iterate through the models and perform GridSearchCV for each
    for model_name, model_info in models.items():
        # Set up GridSearchCV with r2 scoring for each model
        grid_search = GridSearchCV(estimator=model_info['model'], param_grid=model_info['params'],
                                   scoring='r2', cv=5, n_jobs=-1)

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Get the best parameters and R² score for the current model
        best_params = grid_search.best_params_
        current_best_r2 = grid_search.best_score_

        print(f"Best Parameters for {model_name}:", best_params)
        print(f"Best R² Score for {model_name}: {current_best_r2}")

        # If the current model's R² score is better than the best, update the best model
        if abs(current_best_r2) > abs(best_r2):
            best_r2 = current_best_r2
            best_model = grid_search.best_estimator_

    # Print the best model and its R² score
    print("\nBest Model:", best_model)
    print("Best R² Score:", best_r2)

    # Optionally, evaluate on the test set
    y_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    print("Test R² Score:", test_r2)

    y_graph = best_model.predict(X_normalized)
    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot sentiment on the first axis
    ax1.plot(merged_df['date'], merged_df['Price'], color='red', label='Real Price', linewidth=2)
    ax1.set_ylabel('Real Price', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    # Create a third axis for Price_diff (ax3)
    ax3 = ax1.twinx()

    # Position the third axis to the right of ax2 (adjust the position)
#    ax3.spines['right'].set_position(('outward', 60))  # Move ax3 outward for clarity
    ax3.plot(merged_df['date'], y_graph, color='green', label='Pred Price', linewidth=2)
    ax3.set_ylabel('Pred Price', color='green')
    ax3.tick_params(axis='y', labelcolor='green')

    # Title and show the plot
    plt.title('Price and Predicted Price Over Time')
    plt.tight_layout()  # Adjust the plot to avoid overlap
    plt.show()
    plt.savefig('output/latest/myGraph_RobertaNo_' + COIN + '.png')

    # Save the DataFrame to a CSV file
    merged_df.to_csv('output/latest/end_data_' + COIN + '.csv', index=False)

SUBREDDIT_NAME = [ "ethereum" ]
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
#    write_my_file_to_csv()
#    get_analysis()
    get_graphs_and_all()