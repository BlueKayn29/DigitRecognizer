from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np


def split_data(features, output):
    random_indices = np.r
    y_return_train, y_return_test = output[:7 * len(output) // 10], output[7 * len(output) // 10:]
    x_return_train, x_return_test = np.zeros([len(y_return_train), len(features)]), np.zeros(
        [len(y_return_test), len(features)])
    for i in range(len(features)):
        x_return_train[:, i], x_return_test[:, i] = features[i][:7 * len(output) // 10], features[i][
                                                                                         7 * len(output) // 10:]

    return x_return_train, y_return_train, x_return_test, y_return_test
    # return feature[:7 * len(feature) // 10], feature[7 * len(feature) // 10:]


def normalize(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)


def accuracy(l1, l2):
    matches = 0
    for i in range(len(l1)):
        if l1[i] == l2[i]:
            matches += 1
    return (matches / len(l1)) * 100


def ohe_categorical(data, enc_indices=None, encoder=None):
    """parameters: data, enc_indices, encoder\n
       specify enc_indices and encoder if using existing encoder\n
       returns data if encoder given\n
       else returns data, encoder, enc_indices"""
    encoded_data_transposed = []
    dataTransposed = np.transpose(data)
    if enc_indices is None and encoder is None:
        ohe = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore', min_frequency=10)
        enc_indices = []
        for j in range(len(dataTransposed)-1):
            flag = 0
            for element in dataTransposed[j]:
                if type(element) == str:
                    flag = 1
            if flag == 1:
                enc_indices.append(j)
            else:
                mean = np.nanmean(dataTransposed[j])
                for i in range(len(dataTransposed[j])):
                    if dataTransposed[j][i] != dataTransposed[j][i]:
                        dataTransposed[j][i] = mean
                encoded_data_transposed.append(dataTransposed[j])
        encodedColumns = []
        for element in enc_indices:
            encodedColumns.append(dataTransposed[element])
        encodedColumns = np.array(encodedColumns)
        ohe.fit(np.transpose(encodedColumns))
        toAdd = np.transpose(ohe.transform(np.transpose(encodedColumns)))
        for i in range(len(toAdd)):
            encoded_data_transposed.append(toAdd[i])
        encoded_data_transposed.append(dataTransposed[-1])
        return np.transpose(np.array(encoded_data_transposed)), ohe, enc_indices
    else:
        encodedColumns = []
        for element in enc_indices:
            encodedColumns.append(dataTransposed[element])
        encodedColumns = np.array(encodedColumns)
        for j in range(len(dataTransposed)):
            if j not in enc_indices:
                mean = np.nanmean(dataTransposed[j])
                for i in range(len(dataTransposed[j])):
                    if dataTransposed[j][i] != dataTransposed[j][i]:
                        dataTransposed[j][i] = mean
                encoded_data_transposed.append(dataTransposed[j])
        toAdd = np.transpose(encoder.transform(np.transpose(encodedColumns)))
        for i in range(len(toAdd)):
            encoded_data_transposed.append(toAdd[i])

        encoded_data_transposed = np.array(encoded_data_transposed)
        return np.transpose(encoded_data_transposed)
