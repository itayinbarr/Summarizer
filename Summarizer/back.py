# Collect and load the data
import pandas as pd
import tensorflow
# Preprocess the data
from sklearn.preprocessing import StandardScaler
from keras.utils import pad_sequences
import nltk
from nltk.stem import PorterStemmer
from keras.preprocessing.text import Tokenizer

# Split the data into train, val, and test sets
from sklearn.model_selection import train_test_split

# Choose a model and configure it for training
from keras.models import Sequential
from keras.layers import LSTM, Dense

import codecs



def preprocess_pipeline():
    # Step 1: Collect and load the data
    def load_data():
        # Load the data into a Pandas dataframe
        df = pd.read_csv("../data/news_summary.csv", encoding_errors='replace')
        # Split the data into input (X) and output (y) columns
        df = df[['text', 'ctext']]
        df = df.applymap(remove_non_strings)
        print('-load data check!')
        return df

    # Define a function that removes non-string characters
    def remove_non_strings(value):
        # If the value is not a string, return an empty string
        if not isinstance(value, str):
            return ""
        # Otherwise, return the value
        return value

    # Step 2: Preprocess the data
    # ---------------------------
    # Tokenize the text
    # Define a tokenization function
    def tokenize(text):
        # Tokenize the string using the NLTK library
        if not isinstance(text, str):
            text = ''
        tokens = nltk.word_tokenize(text)
        # Return the tokens
        return tokens

    def tokenizing(df):
        tokens1 = [token for sublist in df['text'].apply(tokenize).values.tolist() for token in sublist]
        tokens2 = [token for sublist in df['ctext'].apply(tokenize).values.tolist() for token in sublist]
        tokens = list(set(tokens1).union(set(tokens2)))
        print('-tokenizing data check!')
        return tokens
        # Output: ['This', 'is', 'some', 'text', 'in', 'English', '.']

    # Remove stop words
    def stop_words(tokens):
        stopwords = nltk.corpus.stopwords.words("english")
        filtered_tokens = [token for token in tokens if token not in stopwords]
        print('-stop word data check!')
        return filtered_tokens
        # Output: ['This', 'text', 'English', '.']

    # Perform word stemming
    def stemming(filtered_tokens):
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
        print('-stemming data check!')
        return stemmed_tokens
        # Output: ['thi', 'text', 'english', '.']

    # Encode the text
    def encoder(stemmed_tokens, x, y):
        tokenizer = Tokenizer(lower=True)
        tokenizer.fit_on_texts(stemmed_tokens)
        x_sequences = tokenizer.texts_to_sequences(x)
        y_sequences = tokenizer.texts_to_sequences(y)
        print('-encoding data check!')
        return [x_sequences, y_sequences]
        # Output: [[1], [2], [3], [4]]

    # Tokenize the input data
    def padding(x_sequences,y_sequences):
        # Pad the input sequences to the same length
        max_length = max([len(x) for x in x_sequences])
        x_padded = pad_sequences(x_sequences, maxlen=max_length)

        # Standardize the input data
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_padded)

        # Tokenize and pad the output data
        y_padded = pad_sequences(y_sequences, maxlen=max_length)
        print('-padding data check!')
        return [x_scaled, y_padded, max_length]

    def wrap_up():
        df = load_data()
        tokens = tokenizing(df)
        filtered_tokens = stop_words(tokens)
        stemmed_tokens = stemming(filtered_tokens)
        sequences = encoder(stemmed_tokens, df['ctext'], df['text'])
        padded = padding(sequences[0], sequences[1])
        x_scaled = padded[0]
        y_scaled = padded[1]
        max_length = padded[2]
        return [x_scaled, y_scaled, max_length]

    return wrap_up()


# Step 3: Split the data into train, val, and test sets
def build_model(x_scaled, y_padded, max_length):
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_padded, test_size=0.2)

    # Step 4: Choose a model and configure it for training

    # Define the model
    model = Sequential()
    model.add(LSTM(128, input_shape=(max_length, 1)))
    model.add(Dense(max_length, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Step 5: Train the model and use the validation set to tune the hyperparameters
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

    # Step 6: Evaluate the model on the test set
    score = model.evaluate(x_test, y_test)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    model.save('summary_model')


preprocessed_data = preprocess_pipeline()
build_model(preprocessed_data[0], preprocessed_data[1], preprocessed_data[2])
