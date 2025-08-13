'''
Concept Recap: High-level modules depend on abstractions, not concrete implementations.
ML Context: Your training pipeline depends on a data interface, not a specific CSV or database loader.
'''
class DataLoader:
    def load(self):
        return [[1,2],[3,4]], [0,1]

class Trainer:
    def __init__(self, loader):
        self.loader = loader  # depends on abstraction

    def train(self):
        X, y = self.loader.load()
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(X, y)
        print("Model trained")
        return model

# main
if __name__ == "__main__":
    loader = DataLoader()
    trainer = Trainer(loader)
    trainer.train()
