import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from helper_functions import *
import csv
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


def split_without_zipping(xRaw, yRaw):
    return xRaw[:7 * len(xRaw) // 10], xRaw[7 * len(xRaw) // 10:], yRaw[:7 * len(yRaw) // 10], yRaw[7 * len(yRaw) // 10:]


data = pd.read_csv("train.csv")
data = np.array(data)

y = data[:, 0]
x = data[:, 1:]

x_train, x_test, y_train, y_test = split_without_zipping(x, y)

model = Sequential(
    [
        tf.keras.Input(shape=784, ),
        # Dense(40, activation='relu'),
        # Dense(20, activation='relu'),
        Dense(530, activation='relu'),
        Dense(10, activation='linear'),
    ], name="nine_digit_model"
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

model.fit(x_train, y_train, epochs=10)
logits = model(x_test)
f_x = tf.nn.softmax(logits)

predictions = []
for element in f_x:
    predictions.append(np.argmax(element))
print(accuracy(predictions, y_test))

####################################################
####################################################
data = pd.read_csv("test.csv")
data = np.array(data)
x_test = data
logits = model(x_test)
f_x = tf.nn.softmax(logits)

y_predict_submissions = []
for element in f_x:
    y_predict_submissions.append(np.argmax(element))

filename = "submission_digit_recognizer.csv"
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile, lineterminator='\n')

    # writing the fields
    csvwriter.writerow(["ImageId", "Label"])

    for j in range(len(y_predict_submissions)):
        csvwriter.writerow([j+1, y_predict_submissions[j]])