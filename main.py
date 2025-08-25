import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

def generateAI():
    dataset = pd.read_csv('data.csv')
    dataset = dataset.dropna()

    X = dataset.iloc[:, 1].values.reshape(-1, 1)
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    ai = KNeighborsClassifier(n_neighbors=5)
    ai.fit(X_train, y_train)

    # Save model
    pickle.dump(ai, open('model.pkl', 'wb'))
    print("Model trained and saved as model.pkl")

if __name__ == "__main__":
    generateAI()
