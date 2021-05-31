import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        # binning with cut
        self.dataset['Age'] = pd.cut(self.dataset['Age'], 4)

        # encode labels
        le = LabelEncoder()

        le.fit(self.dataset['Gender'])
        self.dataset['Gender'] = le.transform(self.dataset['Gender'])

        le.fit(self.dataset['Vehicle_Damage'])
        self.dataset['Vehicle_Damage'] = le.transform(self.dataset['Vehicle_Damage'])

        le.fit(self.dataset['Vehicle_Age'])
        self.dataset['Vehicle_Age'] = le.transform(self.dataset['Vehicle_Age'])

        le.fit(self.dataset['Age'])
        self.dataset['Age'] = le.transform(self.dataset['Age'])

        return self.dataset
