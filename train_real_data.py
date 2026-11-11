"""
Meme Scientist - Real Dataset Training Script
Dataset: memegenerator.csv (Library of Congress / Memegenerator Web Archive)
Algorithms: TF-IDF + Multinomial Naive Bayes, Logistic Regression, Random Forest
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_and_evaluate(csv_path="memegenerator.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"--> Loading real meme dataset from: {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Total raw records: {len(df):,}")

    # Clean missing values
    df = df.dropna(subset=["Base Meme Name", "Alternate Text"])
    print(f"Valid records with template and caption: {len(df):,}")

    # Select top 12 most frequent meme templates
    top_templates = df["Base Meme Name"].value_counts().head(12).index.tolist()
    filtered = df[df["Base Meme Name"].isin(top_templates)].copy()
    print(f"\nFiltered down to top 12 meme templates ({len(filtered):,} samples):")
    for rank, (name, count) in enumerate(filtered["Base Meme Name"].value_counts().items(), 1):
        print(f"  {rank:2d}. {name:<35} ({count:,} memes)")

    X = filtered["Alternate Text"]
    y = filtered["Base Meme Name"]

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTraining set size: {len(X_train):,} | Test set size: {len(X_test):,}")

    # Text vectorization using TF-IDF (unigrams + bigrams)
    print("\n--> Vectorizing text using TF-IDF (unigrams & bigrams, max 5,000 features)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 1. Multinomial Naive Bayes
    print("\n[1/3] Training Multinomial Naive Bayes...")
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_train_vec, y_train)
    nb_preds = nb.predict(X_test_vec)
    nb_acc = accuracy_score(y_test, nb_preds)
    print(f"       Multinomial Naive Bayes Accuracy: {nb_acc * 100:.2f}%")

    # 2. Logistic Regression
    print("\n[2/3] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=500, C=1.0)
    lr.fit(X_train_vec, y_train)
    lr_preds = lr.predict(X_test_vec)
    lr_acc = accuracy_score(y_test, lr_preds)
    print(f"       Logistic Regression Accuracy:     {lr_acc * 100:.2f}%")

    # 3. Random Forest Classifier
    print("\n[3/3] Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train_vec, y_train)
    rf_preds = rf.predict(X_test_vec)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"       Random Forest Accuracy:          {rf_acc * 100:.2f}%")

    print("\n=======================================================")
    print("DETAILED CLASSIFICATION REPORT (Logistic Regression)")
    print("=======================================================")
    print(classification_report(y_test, lr_preds, digits=3))

    # Test sample real-world captions
    print("=======================================================")
    print("INFERENCE TESTING ON REAL / CUSTOM CAPTIONS")
    print("=======================================================")
    test_captions = [
        "Not sure if code has bugs or if test suite is broken",
        "Y U NO study for data mining exam",
        "Found twenty dollars in my winter coat pocket",
        "What if sleep is just a trial version of death",
        "I don't always test code but when I do I do it in production"
    ]

    test_vec = vectorizer.transform(test_captions)
    test_preds = lr.predict(test_vec)
    test_probs = lr.predict_proba(test_vec)

    for cap, pred, probs in zip(test_captions, test_preds, test_probs):
        confidence = np.max(probs) * 100
        print(f"Caption   : \"{cap}\"")
        print(f"Predicted : {pred} (Confidence: {confidence:.1f}%)\n")

    print("Training and evaluation completed successfully!")
    return {
        "nb_acc": nb_acc,
        "lr_acc": lr_acc,
        "rf_acc": rf_acc,
        "vectorizer": vectorizer,
        "model": lr,
        "top_templates": top_templates
    }

if __name__ == "__main__":
    train_and_evaluate()

