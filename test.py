import tensorflow as tf
import pandas as pd
import numpy as np

model = tf.keras.models.load_model('train_v2.h5')

test_data = 'test.csv'

new_data = pd.read_csv(test_data)
new_data = new_data.drop('SVMacc', axis=1)
#new_data = new_data.drop('y_class', axis=1)


new_data = new_data.values
new_data = new_data.reshape(new_data.shape[0], new_data.shape[1], 1) 

predictions = model.predict(new_data)
predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes)

# 자세한 클래스별 예측값 출력
print(predictions[0])
print(predictions[1])
print(predictions[2])
print(predictions[3])
# ...

