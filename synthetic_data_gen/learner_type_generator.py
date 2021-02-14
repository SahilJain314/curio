import pandas as pd
import random

class LearnerTypeGenerator:

    def __init__(self, feature_dataset):
        self.incidents = pd.read_csv(feature_dataset)
        self.csv_path = feature_dataset

    def update_file(self):
        self.incidents.to_csv(self.csv_path, mode="w", header=True, index=False)
    
    def generate_new_person_data(self):
        # print(self.incidents.loc[1, :])
        seed_index = random.randint(1, len(self.incidents) - 1)
        data = self.incidents.loc[seed_index, :].copy()
        rand_incidents_base_index = 3*(random.randint(1, 20) - 1)
        question_choice_index = random.randint(0, 2)
        ans_choice = [0, 0, 0]
        ans_choice[question_choice_index] = 1
        data[rand_incidents_base_index:rand_incidents_base_index+3] = ans_choice
        data[0] = len(self.incidents) + 1
        print(len(self.incidents))
        self.incidents = self.incidents.append(data, ignore_index=True)
        return data
