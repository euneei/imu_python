import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


class Model:
    def __init__(self, data_path):
        self.data_path = data_path        
        self.model = None

    def load_data(self, apply_fft=False, sampling_freq=50):
        df = pd.read_csv(self.data_path)
        df = df.drop('SVMacc', axis=1)
        X = df.drop('y_class', axis=1).values
        y = df['y_class'].values
        
        if apply_fft:
            # 데이터 X에 대해 각 행별로 FFT 실행
            fft_values = np.fft.fft(X, axis=1)  
            fft_freq = np.fft.fftfreq(X.shape[1], 1 / sampling_freq)  
            pos_indices = fft_freq > 0
            X = np.abs(fft_values[:, pos_indices]) / X.shape[1]  # Magnitude only

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )
        return self.X_train, self.X_val, self.y_train, self.y_val

    def train_model(self):
        self.model = tf.keras.Sequential([
        
        
            tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(300,1)),
            tf.keras.layers.Conv1D(filters=8, kernel_size=5, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(self.X_train, self.y_train, epochs=50, batch_size=10, validation_data=(self.X_val, self.y_val))
        self.model.save('traincnn.h5')

class ModelTest:
    def __init__(self, model_path, test_data_path):
        self.model_path = model_path
        self.test_data_path = test_data_path

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)

    def evaluate_model(self):
        new_data = pd.read_csv(self.test_data_path)
        new_data = new_data.drop('SVMacc', axis=1).values
        new_data = new_data.reshape(new_data.shape[0], new_data.shape[1], 1)
        predictions = self.model.predict(new_data)
        predicted_classes = np.argmax(predictions, axis=1)
        return predictions, predicted_classes
    

if __name__ == "__main__":
    trainer = Model('data221(walk,run,oth).csv')
    trainer.load_data(apply_fft=True)  # FFT를 적용하려면 True로 설정
    trainer.train_model()

    tester = ModelTest('traincnn.h5', '.../test.csv')
    tester.load_model()
    predictions, predicted_classes = tester.evaluate_model()
    print(predicted_classes)
    print(predictions[0])
    print(predictions[1])
    print(predictions[2])
    print(predictions[3])
