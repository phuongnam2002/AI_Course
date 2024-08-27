import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from preprocess_data import (
    preprocessing,
    convert_text_to_vector,
    convert_text_to_label
)

if __name__ == '__main__':
    df_twitter = pd.read_csv('data/raw_data/cleaned_twitter_data.csv')
    df_guardian = pd.read_csv('data/raw_data/cleaned_guardian_df.csv')

    inputs_twitter = df_twitter['text'].values.tolist()
    labels_twitter = df_twitter['sentiment'].values.tolist()

    inputs_guardian = df_guardian['text'].values.tolist()
    labels_guardian = df_guardian['sentiment'].values.tolist()

    inputs = inputs_twitter + inputs_guardian
    labels = labels_twitter + labels_guardian

    inputs = list(map(preprocessing, inputs))

    data = convert_text_to_vector(inputs)
    labels = convert_text_to_label(labels)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=10000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("\nClassification Report:\n", report)
