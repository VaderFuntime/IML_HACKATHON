from sklearn.linear_model import RidgeCV
import numpy as np

"""
A model that predicts a distribution table of appearances of event types in each time slot using ridge regression
"""

class DistributionPredictor:
    def __init__(self):
        self.model = RidgeCV(alphas=np.linspace(0.1, 100, 20), cv=10)
        self.events = ["ACCIDENT", "JAM", "ROAD_CLOSED", "WEATHERHAZARD"]

    """
    Fit the ridge regression model over design matrix X and labels y
    """

    def fit(self, X, y):
        self.model = self.model.fit(X, np.log(y))
        return self

    """
    Predict function, for each valid time slot and for each event, predicts how many times the event will appear in
    the given time slot on the given day of week parameter
    """

    def predict(self, dayofweek):
        preds = np.array([[0] * len(self.events)] * 3)
        for j, time_slot in enumerate([1, 2, 3]):
            for i in range(len(self.events)):
                sample = [time_slot, dayofweek, 0, 0, 0, 0]
                sample[i + 2] = 1
                pred = self.model.predict(np.array(sample).reshape(1, -1))
                preds[j][i] = np.exp(pred)

        return preds
