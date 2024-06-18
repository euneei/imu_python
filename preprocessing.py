import pandas as pd
import glob
import os

class Preprocessor:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def read_and_label(self, file_path, label):
        df = pd.read_csv(file_path, sep='\t', header=None, names=['n', 'x', 'y', 'z'])
        df['SVMacc'] = (df['x']**2 + df['y']**2 + df['z']**2)**0.5
        df['SVMacc'] = df['SVMacc'].round(6)
        df = df[['SVMacc']]
        df = df.transpose()
        df.insert(0, 'y_class', label)
        return df

    def combine_data(self, pattern, label):
        file_list = glob.glob(os.path.join(self.base_dir, pattern))
        dataframes = [self.read_and_label(file_path, label) for file_path in file_list]
        combined_df = pd.concat(dataframes)
        return combined_df

    def process_all_data(self, patterns_labels):
        all_dataframes = []
        for pattern, label in patterns_labels.items():
            df = self.combine_data(pattern, label)
            all_dataframes.append(df)
        final_df = pd.concat(all_dataframes)
        final_df.to_csv(os.path.join(self.base_dir, 'data.csv'))
