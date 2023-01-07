Summarizer
==============================

This is a model trained to summarize an article to a few sentences.

This project demonstrates the process of NLP preprocessing pipeline of raw data, from
removing the stop words to tokenizing and using the processed data to train a LSTM model.

It's important to say that this specific model would be suitable for this kind of job. I would choose sec2sec model
instead. My main focus here was the NLP preprocessing.


Getting Started
------------

From within the repo directory run

`./Summarizer/back.py`

You can now see the accuracy level of the model.

-----
About Training & Dataset
--

The dataset was derived from Kaggle. It consists of thousands of news articles and their short summaries.

Project Organization
------------

    ├── README.md                    <- The top-level README for developers using this project
    ├── LICENSE.md                   <- MIT
    ├── .gitignore                   <- For environment directories
    │
    ├── Summarizer                   <- Containing the software itself
    │   ├── back.py                  <- backend code
    │
    └── tests                        <- Tests directory, .gitignored
        └── backend_tests.py         <- Unit tests of backend
 
Dependencies
------------

- Python
- NLTK
- scikit-learn
- Pandas
- NumPy
--------
# Summarizer
# Summarizer
