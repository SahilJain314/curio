import pandas as pd
import numpy.random as random
import json
import learner_type_generator as ltg

class VideoDataGenerator:

    def __init__(self, data_path, learner_feature_data):
        self.incidents = pd.read_csv(data_path)
        self.data_path = data_path
        self.learner_feature_data = learner_feature_data

    def update_file(self):
        self.incidents.to_csv(self.data_path, mode="w", header=True, index=False)

    def generate_video_affinity_data(self, player_id):
        with open("../youtube_video/full_data.json") as f:
            full_data = json.load(f)
        # feature_dataset = pd.read_csv(learner_feature_data) # gives us their learning style (0, 1, 2)
        
        generator = ltg.LearnerTypeGenerator(self.learner_feature_data)
        video_list = []
        learning_type = generator.dot_product(player_id)

        for topic in full_data.keys():
            for i in range(len(full_data[topic])):
                if i == learning_type:
                    norm = random.normal(4, 1)
                    affinity = (norm < 1)*1.0 + (norm >= 2 and norm <=5)*norm + (norm>5)*5.0
                    
                    # print((player_id, full_data[topic][i]["videoId"], int(affinity)))
                    video_list.append((player_id, full_data[topic][i]["videoId"], int(affinity)))
                    break
            video_index = random.randint(4, 9)
            denorm = random.normal(1, 3)
            diffinity = (denorm < 0.0)*0.0 + (denorm >= 0.0 and denorm <= 5.0)*denorm + (denorm>5.0)*5.0
            video_list.append((player_id, full_data[topic][video_index]["videoId"], int(-1*diffinity)))
        
        
        self.incidents = self.incidents.append(video_list, ignore_index=True)
        return video_list
