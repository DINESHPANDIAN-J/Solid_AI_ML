'''
Concept Recap: Code should be open for extension, closed for modification.
ML Context: Add new evaluation metrics without modifying existing evaluator logic.
'''

from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, f1_score

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, y_true, y_pred):
        pass

class AccuracyEvaluator(Evaluator):
    def evaluate(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

class F1Evaluator(Evaluator):
    def evaluate(self, y_true, y_pred):
        return f1_score(y_true, y_pred)

# Hands-on usage
if __name__ == "__main__":
    y_true = [0,1,0,1,1]
    y_pred = [0,1,1,1,0]

    for evaluator in [AccuracyEvaluator(), F1Evaluator()]:
        print(f"{evaluator.__class__.__name__}: {evaluator.evaluate(y_true, y_pred)}")
