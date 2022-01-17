import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
import tensorflow as tf
from tensorflow.keras import layers
import re, string
import pickle

#import nltk
#nltk.download("stopwords")
MAXLEN = 35
VOCAB_SIZE = 10000

def save_tokenizer(tokenizer):
    """save the tokenizer for future use"""
    import pickle
    with open('src/models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer():
    with open('src/models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def shorten_dataset(data, n=5000):
    """ Shuffle and shorten the dataset in case you want to test """
    data = data.sample(frac=1)
    data = data.head(n)
    return data

def change_sentiments(data):
    """ The sentiment from the input data is 0 for unhappy, 4 for happy. Let's change the 4 to 1 """
    data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x==4 else 0)
    return data

def clean_tweet(tweet:str):
    """ Removing punctuation, hashtags, lowercasing everything. The link remover needs to be fixed, as it currently deletes every word after the link """
    tweet = tweet.lower()
    #tweet = re.sub(r'https?:\/\/.*[\r\n]* ', '', str(tweet)) #TODO: fix this link remover, it currently deletes everything beyond the link
    tweet = re.sub(r'#', '', str(tweet)) #remove hashtab

    #remove punctuation
    punct = set(string.punctuation)
    tweet = "".join(ch for ch in tweet if ch not in punct)
    return tweet

def make_numpy(in_data):
    """turns the relevant columns in pandas dataframe into numpy arrays"""
    tweets = in_data['tweet'].to_numpy()
    sentiments = in_data['sentiment'].to_numpy()
    return tweets, sentiments

def seq_padder(tokenizer, tweets):
    """tokenizes and pads the tweets"""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    sequences = tokenizer.texts_to_sequences(tweets)
    print(sequences[0])
    padded = pad_sequences(sequences, truncating='post', padding='post', maxlen=MAXLEN)
    return padded

def make_tokenizer(vocab_size):
    """ make the tokenizer """
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<UNK>')
    tokenizer.fit_on_texts(tweets)
    return tokenizer

def create_model():
    """ Create a bidirectional LSTM model for sentiment analysis """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 16, input_length=MAXLEN),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(40, return_sequences=True)),
        layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(40)),
        layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def batch_data(X_train, y_train, X_test, y_test):
    """Shuffle and batch the training and test data"""
    BATCH_SIZE = 512
    BUFFER_SIZE = 10000
    dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dataset_train = dataset_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    dataset_test = dataset_test.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    return dataset_train, dataset_test

if __name__ == "__main__":
    data = pd.read_csv('data/tweets.csv', encoding = 'Latin-1', names=('sentiment','id','date','flag','username','tweet'))

    #data = shorten_dataset(data, 50000)
    #pred = make_prediction([[6, 148, 72, 125, 40, 0.627, 50]])
    #print("The test prediction is {}".format(pred))

#print(model.predict_proba(transformed_example))
