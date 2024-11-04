# This is a sample Python script.
from itertools import count

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import zstandard
import json
from datetime import datetime
import re
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

FIRST_TRY = True
SECOND_TRY = False

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
    "Bitcoin_comments",
    "Bitcoin_submissions",
    "ethereum_comments",
    "ethereum_submissions"
]

def write_my_file_to_csv():
    for input_output_file in INPUT_OUTPUT_FILES:
        input_path = 'subreddits23/' + input_output_file + '.zst'
        output_path = 'output/' + input_output_file + '_clean_reduced_final.csv'
        start_date = datetime.strptime("2021-01-01", "%Y-%m-%d")
        final_date = datetime.strptime("2022-11-30", "%Y-%m-%d")

        desired_keys = [
            #'author',
            'created_utc',
            #'num_comments', 'pwls', 'score', 'selftext',
            'subreddit',
            'title',
            #'upvote_ratio', 'wls', 'controversiality',
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
        cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)  # Remove punctuation and symbols
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Remove extra whitespace
        return cleaned_text.strip()  # Remove leading/trailing spaces
    return text  # Return original if not a string

def get_sentiment(text):
    if isinstance(text, str):  # Check if the input is a string
        return TextBlob(text).sentiment.polarity
    else:
        return None  # Return None for non-string inputs

def get_analysis():
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    for COMMON_NAME in SUBREDDIT_NAME:
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
        combined_df = pd.concat([submissions_df, comments_df], ignore_index=True)

        print(f'Start Number of elements in the combined DataFrame: {len(combined_df)}')

        # Handle potential mixed types in 'created_utc'
        combined_df['created_utc'] = pd.to_numeric(combined_df['created_utc'], errors='coerce')

        tqdm.pandas(desc="Processing")
        combined_df['body'] = combined_df['body'].progress_apply(spell_check.replace_url)

        # Drop rows where created_utc is NaN (if any)
        combined_df = combined_df.dropna(subset=['created_utc'])

        # Sort the combined DataFrame by 'created_utc'
        combined_df = combined_df.sort_values(by='created_utc')

        combined_df['sentiment'] = combined_df['body'].progress_apply(get_sentiment)

        # Save the combined DataFrame to a new CSV file
        combined_df.to_csv('output/E_' + COMMON_NAME + '_combined_final.csv', index=False)
        print(f'End Number of elements in the combined DataFrame: {len(combined_df)}')


def get_graphs_and_all():
#    for COIN in SUBREDDIT_NAME:
#        print(f"Loading {COIN} in csv")
    COIN = 'twitter'
    NOT_PROCESSED = False
    # Load data
    if NOT_PROCESSED:
        df = pd.read_csv("output/latest/E_" + COIN + "_combined_final.csv")
        price_df = pd.read_csv("cryptoPrices/" + COIN + "_price.csv")

        # Filter sentiments
        df = df[(df['sentiment'] > 0.5) | (df['sentiment'] < -0.5)]

        # Prepare date and average sentiment
        df['date'] = pd.to_datetime(df['created_utc'], unit='s').dt.date
        df['date'] = pd.to_datetime(df['date'])

        # Set the 'date' column as the index for grouping
        df.set_index('date', inplace=True)

        # Group by week and calculate average sentiment
        sentiment_df = df.resample('W')['sentiment'].mean().reset_index()
        sentiment_df.columns = ['date', 'average_sentiment']

        # Reset the index of sentiment_df to merge properly
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

        # Merge price and sentiment DataFrames
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        merged_df = pd.merge(price_df, sentiment_df, left_on='Date', right_on='date', how='outer')

        merged_df = merged_df[['Date', 'Price', 'average_sentiment']]

    else:
        merged_df = pd.read_csv('output/latest/E_twitter_combined_final.csv')

    # Drop rows with missing values
    merged_df = merged_df.dropna(subset=['avg_sentiment', 'Price'])
#    merged_df['Price'] = merged_df['Price'].str.replace(',', '').astype(float)

    merged_df.to_csv("output/latest/merged_dataframes_" + COIN + ".csv")

    merged_df['date'] = pd.to_datetime(merged_df['date'])

    # Calculate the difference in price
    merged_df['Price_Diff'] = merged_df['Price'].diff()

    # Drop the first row which will have a NaN value in Price_Diff
    merged_df = merged_df.dropna()

    # Prepare the features and target variable
    X = merged_df[['avg_sentiment']]
    y = merged_df['Price']

    # Normalize the data
#    scaler = MinMaxScaler()
#    X_normalized = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.show()
    plt.savefig('output/latest/myGraph_' + COIN + '.png')

    print(f'MSE: {mse}, MAE: {mae}, R^2: {r2}')
    # Save the DataFrame to a CSV file
    merged_df.to_csv('output/latest/end_data_' + COIN + '.csv', index=False)

SUBREDDIT_NAME = [ "ethereum", "Bitcoin" ]
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
#    write_my_file_to_csv()
#    get_analysis()
    get_graphs_and_all()