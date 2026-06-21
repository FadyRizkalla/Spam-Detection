# =============================================================================
# COMP 472 - Artificial Intelligence | Summer 2026
# Mini Project 2 - Spam Detection AI
# =============================================================================
# Libraries used:
#   pandas       - dataset loading and manipulation
#   scikit-learn - TF-IDF and model training
# =============================================================================

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# =============================================================================
# 1. LOAD DATASET
# =============================================================================

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the SMS Spam Collection CSV file.

    The dataset has a 'lable' column (note: typo in original file) and
    a 'message' column. Extra unnamed columns are dropped.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns ['label', 'message'].

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing.
    """
    try:
        df = pd.read_csv(filepath, encoding='latin-1')
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    # Keep only the first two columns regardless of header name
    df = df.iloc[:, :2]
    df.columns = ['label', 'message']

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Validate labels
    valid_labels = {'spam', 'ham'}
    if not set(df['label'].unique()).issubset(valid_labels):
        raise ValueError(f"Unexpected labels found: {df['label'].unique()}")

    return df


# =============================================================================
# 2. FEATURE EXTRACTION  (TF-IDF)
# =============================================================================

def extract_features(X_train: pd.Series, X_test: pd.Series):
    """
    Convert raw email/SMS text into numerical TF-IDF feature vectors.

    TF-IDF (Term Frequency - Inverse Document Frequency):
      - TF  : how often a word appears in a message.
      - IDF : penalises words that appear in almost every message (e.g. "the").
      - Together they highlight words that are distinctive for a message.

    Args:
        X_train: Training messages (Series of strings).
        X_test:  Testing messages  (Series of strings).

    Returns:
        Tuple (X_train_tfidf, X_test_tfidf, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        stop_words='english',   # ignore common English words
        max_features=5000,      # keep the 5000 most informative words
        ngram_range=(1, 2)      # use single words AND two-word phrases
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)   # learn vocabulary + transform
    X_test_tfidf  = vectorizer.transform(X_test)        # transform only (no re-learning)

    return X_train_tfidf, X_test_tfidf, vectorizer


# =============================================================================
# 3. MODEL TRAINING
# =============================================================================

def train_model(X_train_tfidf, y_train: pd.Series) -> MultinomialNB:
    """
    Train a Multinomial Naive Bayes classifier.

    Naive Bayes works well for text classification because:
      - It assumes each word contributes independently to the probability.
      - It is fast and performs surprisingly well even with this simplification.
      - MultinomialNB is designed for discrete counts / TF-IDF weights.

    Args:
        X_train_tfidf: Sparse TF-IDF matrix for training data.
        y_train:       Training labels ('spam' / 'ham').

    Returns:
        Trained MultinomialNB model.
    """
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return model


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 45)
    print("     Welcome to Spam Detection AI")
    print("=" * 45)

    # ── 1. Load dataset ───────────────────────────────
    dataset_path = "spam (1).csv"   # place the CSV in the same folder as this script
    print(f"\nLoading dataset from '{dataset_path}' ...")
    df = load_dataset(dataset_path)
    print(f"Dataset loaded: {len(df)} messages  "
          f"({df['label'].value_counts()['spam']} spam, "
          f"{df['label'].value_counts()['ham']} ham)")

    # ── 2. Split into training / testing sets ─────────
    #   80 % for training, 20 % for testing
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"\nTraining samples : {len(X_train)}")
    print(f"Testing  samples : {len(X_test)}")

    # ── 3. Feature extraction (TF-IDF) ────────────────
    print("\nExtracting TF-IDF features ...")
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)
    print(f"Vocabulary size  : {len(vectorizer.vocabulary_)} features")

    # ── 4. Train model ────────────────────────────────
    print("\nTraining model ...")
    model = train_model(X_train_tfidf, y_train)
    print("Model training complete.")


if __name__ == "__main__":
    main()
