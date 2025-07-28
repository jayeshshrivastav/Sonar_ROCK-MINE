import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os


def split(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )
    return X_train ,X_test,y_train,y_test

def CusTrainer(X_train,X_test,Y_train,Y_test):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, Y_train)
    train_acc = accuracy_score(Y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(Y_test, model.predict(X_test_scaled))

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    os.makedirs("model", exist_ok=True)
    with open("model/sonar_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        return (f"Model saved successfully : {f}")
    

if __name__=="__main__":

    dataF = pd.read_csv("Dataset/sonar_data.csv", header=None)
    XTR,XTS,YTR,YTS=split(dataF)
    CusTrainer(X_train=XTR,X_test=XTS,Y_train=YTR,Y_test=YTS)
    print("SF")