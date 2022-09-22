import collections
from typing import *
import numpy as np
import pandas as pd


class TypeProb:
    SEP = ","

    def __init__(self, subtypes: str):
        self.subtypes = subtypes
        self.probs = dict()
        self.word_count = 0

    def add_sub(self, subtype: str):
        if subtype in self.probs:
            self.probs[subtype] += 1
        else:
            self.probs[subtype] = 1
        self.word_count += 1

    def get_most_likely(self):
        return max(self.probs.keys(), key=lambda k: self.probs[k])

    def get_prob(self, subtype):
        if subtype not in self.probs:
            return 0
        return self.probs[subtype]/self.word_count

    @staticmethod
    def deque_to_str(d: collections.deque, start=0, end=4):
        s = ""
        for i in range(start, end):
            s += str(d[i])
            if i < end-1:
                s += TypeProb.SEP
        return s