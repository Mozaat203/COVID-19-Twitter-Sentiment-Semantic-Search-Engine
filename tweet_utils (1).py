import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    tweet = re.sub(r'\$\w*', '', tweet)          # remove stock tickers
    tweet = re.sub(r'^RT[\s]+', '', tweet)       # remove RT
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)  # remove URLs
    # tweet = re.sub(r'#', '', tweet)            # ❌ REMOVED

    tokenizer = TweetTokenizer(
        preserve_case=False,
        strip_handles=True,
        reduce_len=True
    )

    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if word not in stopwords_english and word not in string.punctuation:
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean
