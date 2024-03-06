import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

nltk.download("stopwords")
nltk.download("wordnet")


def preprocess_text_data(input_df: pd.DataFrame, txt_col: str) -> pd.DataFrame:
    """This function takes the raw input CSV file and performs the following pre-processing steps
    on the inputs.

    1. Cleaning text data - Remove unnecessary characters, special symbols, and numbers.
    2. Lowercasing - Convert all text to lowercase to ensure consistency.
    3. Tokenization - Split sentences into individual words or tokens.
    4. Removing stopwords - Remove common words that do not contribute much to sentiment.
    5. Lemmatization or Stemming - Reduce words to their base or root form.

    Returns:
        pd.DataFrame: Processed pandas dataframe.
    """

    # Cleaning
    input_df["cleaned_text_new"] = input_df[txt_col].str.replace("[^a-zA-Z\s]", "")

    # Lowercasing
    input_df["cleaned_text_new"] = input_df[txt_col].str.lower()

    # Tokenization
    input_df["tokens"] = input_df["cleaned_text_new"].apply(word_tokenize)

    # Removing stopwords
    stop_words = set(stopwords.words("english"))
    input_df["tokens"] = input_df["tokens"].apply(
        lambda x: [word for word in x if word not in stop_words]
    )

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    input_df["tokens"] = input_df["tokens"].apply(
        lambda x: [lemmatizer.lemmatize(word) for word in x]
    )

    return input_df


if __name__ == "__main__":
    df1 = pd.read_csv("../data/raw/depression_dataset_reddit_twitter.csv")
    df2 = pd.read_csv(
        "../data/raw/data_collection_v1.0.2.xlsx - data_collection_v1.0.2.csv"
    )

    res1 = preprocess_text_data(df1, "clean_text")
    res2 = preprocess_text_data(df2, "text")
