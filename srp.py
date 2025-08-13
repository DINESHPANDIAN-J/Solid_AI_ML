'''
Module 1: Single Responsibility Principle (SRP)
Concept Recap: Each class should do one thing only.
ML Context: Preprocessing, feature engineering, model training, and evaluation are all separate responsibilities.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Data Preprocessing
class DataPreprocessor:
    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        return df

    def preprocess(self, df):
        # fill missing age values
        df['Age'].fillna(df['Age'].mean(), inplace=True)
        df['Sex'] = df['Sex'].map({'male':0, 'female':1})
        return df[['Pclass','Sex','Age','Survived']], df['Survived']

# Model Training
class ModelTrainer:
    def train(self, X_train, y_train):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model

# main
if __name__ == "__main__":
    data_file = "data/titanic.csv"
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(preprocessor.load_data(data_file))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = ModelTrainer().train(X_train, y_train)
    print("Model trained successfully!")
