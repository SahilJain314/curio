import pandas as pd
import learner_type_generator as ltg
import video_data_generator as vdg
from operator import itemgetter

if __name__ == "__main__":
    dataset = "learner_feature_data.csv"
    generator = ltg.LearnerTypeGenerator(dataset)
    vid_gen = vdg.VideoDataGenerator("video_data.csv", "learner_feature_data.csv")
    print(vid_gen.generate_video_affinity_data(1))
    vid_gen.update_file()
    # for i in range(0, 200):
    #     generator.generate_new_person_data()
    # generator.update_file()

    # print(generator.dot_product(4))
    # print(max(generator.dot_product(2), key=itemgetter(0))[0])
    # print(generator.dot_product(3))
    # print(generator.dot_product(4))