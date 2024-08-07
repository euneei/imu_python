import tensorflow as tf
import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
      

    def svm_process(self):
        try: 
            df = pd.read_csv(file_path, sep='\t', header=None, names=['n', 'x', 'y', 'z'])
            df['SVMacc'] = (df['x']**2 + df['y']**2 + df['z']**2)**0.5
            df['SVMacc'] = df['SVMacc'].round(6)
            df.drop(['n', 'x', 'y', 'z'], axis=1, inplace=True)
            df = df.transpose()
            return df
        except Exception as e:
            print(f"Failed to read or process data: {e}")
            return None
        
    def overlap_df(self, data, window_size = 300, step_size=300):

        if data is not None:

            data = data.values.reshape(-1,1)
            data = pd.DataFrame(data, columns=['value'])


            slicing_window = []
            print("slicing_window=[]")
            for start in range(0,len(data)-window_size+1, step_size):
                end = start+window_size
                window = data.iloc[start:end].values.flatten()
                slicing_window.append(window)
            
            new_df = pd.DataFrame(slicing_window)
            print(new_df)
            return new_df
        else:
            print("Data is not available")
            return None

file_path = 'acc_01.txt'

processor = DataProcessor(file_path)
processed_data = processor.svm_process()
print(processed_data)

new_data = processor.overlap_df(processed_data)
#new_data = new_data.drop('SVMacc', axis=1)
if new_data is not None:
    print(new_data.shape)
else:
    print("Processing data failed.")
    exit()

new_data = new_data.values
new_data = new_data.reshape(new_data.shape[0],new_data.shape[1],1)



 # 3. 테스트
model = tf.keras.models.load_model('train_v2.h5')


predictions_df1 = model.predict(new_data)
predicted_classes_df1 = np.argmax(predictions_df1, axis=1)
print(predicted_classes_df1)
