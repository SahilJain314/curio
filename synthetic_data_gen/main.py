import pandas as pd
import learner_type_generator as ltg

if __name__ == "__main__":
    dataset = "learner_feature_data.csv"
    generator = ltg.LearnerTypeGenerator(dataset)

    for i in range(0, 100):
        generator.generate_new_person_data()
    generator.update_file()