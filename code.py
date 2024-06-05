import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def load_data(file_path):
    return pd.read_excel(file_path)

def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)
    df['implied_probability'] = 1 / df['odds']
    df['binary_outcome'] = df['outcome'].apply(lambda x: 1 if x == 'win' else 0)
    return df

def perform_eda(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['odds'], bins=30, kde=True)
    plt.title('Distribution of Betting Odds')
    plt.xlabel('Odds')
    plt.ylabel('Frequency')
    plt.savefig('odds_distribution.png')

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='odds', y='binary_outcome', data=df)
    plt.title('Odds vs Outcome')
    plt.xlabel('Odds')
    plt.ylabel('Outcome')
    plt.savefig('odds_vs_outcome.png')

    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')

def train_model(df):
    X = df[['odds', 'implied_probability']]
    y = df['binary_outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    with open('classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))

    confusion = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')

    return model

def extract_insights(model, df):
    coefficients = pd.DataFrame({
        'Feature': ['odds', 'implied_probability'],
        'Coefficient': model.coef_[0]
    })
    coefficients.to_csv('model_coefficients.csv', index=False)

    odds_bins = pd.cut(df['odds'], bins=[1, 2, 3, 4, 5, 10, np.inf])
    outcomes_by_odds = df.groupby(odds_bins)['binary_outcome'].mean()
    outcomes_by_odds.to_csv('outcomes_by_odds.csv')

    plt.figure(figsize=(10, 6))
    sns.barplot(x=outcomes_by_odds.index.astype(str), y=outcomes_by_odds.values)
    plt.title('Outcome Probability by Odds Range')
    plt.xlabel('Odds Range')
    plt.ylabel('Win Probability')
    plt.savefig('outcome_probability_by_odds.png')

def main(file_path):
    df = load_data(file_path)
    df = preprocess_data(df)
    perform_eda(df)
    model = train_model(df)
    extract_insights(model, df)

#runs our code without needing any manual input.
if __name__ == '__main__':
    # Replace 'your_betting_data.xlsx' with the path to your actual Excel file
    main('your_betting_data.xlsx')
