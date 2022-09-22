from collections import deque
import plotly.graph_objects as go
import pandas as pd
from type_prob_class import TypeProb
import matplotlib.pyplot as plt
import plotly.express as px

"""
Predicator of next event type given training set
"""
class SubtypePredictor:

    def __init__(self):
        self.prob_list = dict()
        self.prefix = {"HAZARD": "WEATHERHAZARD", "JAM": "JAM", "ROAD": "ROAD_CLOSED", "ACCIDENT": "ACCIDENT"}

    """
    Get type of given subtype
    """
    def sub_to_type(self, subtype: str):
        for k in self.prefix:
            if subtype.startswith(k):
                return self.prefix[k]
        raise Exception("Subtype " + subtype + " has no prefix in dictionary!")

    """
    Create probability lists for sequences of event in given length 
    """
    def add_prob_lists(self, df: pd.DataFrame, queue_size, reset_per_city=True):
        curr_city = ""
        sequence = deque()

        for index, sample in df.iterrows():
            city = sample["city"]
            if reset_per_city and (not curr_city or curr_city != city):
                sequence.clear()
                curr_city = city
            sub = str(sample["subtype"])

            if len(sequence) < queue_size:
                sequence.append(sub)
                continue

            curr_word = TypeProb.deque_to_str(sequence, 0, queue_size)
            if curr_word not in self.prob_list:
                self.prob_list[curr_word] = TypeProb(curr_word)

            probs = self.prob_list[curr_word]
            probs.add_sub(sub)
            sequence.append(sub)
            sequence.popleft()

        return self.prob_list

    """
    Given a training set, create probability lists for all occurring sequences
    """
    def fit(self, train: pd.DataFrame):
        self.add_prob_lists(train, 4, False)
        self.add_prob_lists(train, 3, False)

    """
    Given a test set, for each sequence of four events, predict the next event according to the event
    with the highest probability from the probability list
    """
    def predict(self, test: pd.DataFrame):
        subs = []
        for sample in range(0, len(test), 4):
            samples = test[sample: sample + 4]
            samples = deque(samples["subtype"].tolist())
            curr_word = TypeProb.deque_to_str(samples)
            if not (curr_word in self.prob_list):
                curr_word = TypeProb.deque_to_str(samples, 1)
            subs.append(self.prob_list[curr_word].get_most_likely())

        data = [[self.sub_to_type(sub), sub] for sub in subs]
        return pd.DataFrame(data, columns=['type', 'subtype'])

    """
    Flex
    """
    def view_bar_plot(self):
        temp_df = pd.DataFrame()

        seq = 'JAM_STAND_STILL_TRAFFIC,JAM_STAND_STILL_TRAFFIC,JAM_STAND_STILL_TRAFFIC,JAM_HEAVY_TRAFFIC'
        words = self.prob_list[seq].probs.keys()
        vals = self.prob_list[seq].probs.values()
        temp_df["Events"] = words
        temp_df["Probability"] = vals
        fig = px.bar(temp_df, x="Events",y="Probability", title="Likelihood of next possible event after sequence:<br>"+seq.replace(","," -> "))
        fig.show()