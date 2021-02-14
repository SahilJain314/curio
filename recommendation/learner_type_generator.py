import pandas as pd
import random
import numpy as np

class LearnerTypeGenerator:

    def __init__(self, feature_dataset):
        self.incidents = pd.read_csv(feature_dataset)
        self.csv_path = feature_dataset

    def update_file(self):
        self.incidents.to_csv(self.csv_path, mode="w", header=True, index=False)
    
    def generate_new_person_data(self):
        seed_index = random.randint(0, len(self.incidents) - 1)
        data = self.incidents.loc[seed_index, :].copy()
        rand_incidents_base_index = 3*(random.randint(1, 20) - 1)+1
        question_choice_index = random.randint(0, 11) % 3
        ans_choice = [0, 0, 0]
        ans_choice[question_choice_index] = 1
        data[rand_incidents_base_index:rand_incidents_base_index+3] = ans_choice
        data[0] = len(self.incidents) + 1
        self.incidents = self.incidents.append(data, ignore_index=True)
        return data

    def dot_product(self, person_id):
        person_info = self.incidents.loc[self.incidents['person_id']==person_id]
        nppi = np.array(person_info)
        a = np.array(self.incidents.loc[self.incidents['person_id']==1])
        b = np.array(self.incidents.loc[self.incidents['person_id']==2])
        c = np.array(self.incidents.loc[self.incidents['person_id']==3])
        dota = nppi[0][1:] * a[0][1:]
        dotb = nppi[0][1:] * b[0][1:]
        dotc = nppi[0][1:] * c[0][1:]
        ans = [dota.sum()/20, dotb.sum()/20, dotc.sum()/20]
        return ans.index(max(ans))

    def get_number_person(self):
        return len(self.incidents)
