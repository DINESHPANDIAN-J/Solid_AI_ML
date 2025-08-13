'''
Concept Recap: Subclasses should replace base classes without breaking code.
ML Context: Different ML models should be interchangeable in pipelines.
'''
class BaseModel:
    def train(self, X, y):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError

class LogisticModel(BaseModel):
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class TreeModel(BaseModel):
    def __init__(self):
        from sklearn.tree import DecisionTreeClassifier
        self.model = DecisionTreeClassifier()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# main
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = X[:100], X[100:], y[:100], y[100:]

    models = [LogisticModel(), TreeModel()]
    for m in models:
        m.train(X_train, y_train)
        print(f"{m.__class__.__name__} predictions: {m.predict(X_test[:5])}")