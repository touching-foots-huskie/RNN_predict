# : Harvey Chang
# : chnme40cs@gmail.com
# this file is used to define the keras model for prediction
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from plant import data_gen
import numpy as np
from matplotlib import pyplot as plt


def sequence_data(X, Y, look_up):
    # pass test
    datax, datay = [], []
    for j in range(Y.shape[0] - look_up):
        datax.append(np.asmatrix(X[j:j + look_up]).T)
        datay.append(Y[j + look_up])
    datax = np.asarray(datax, dtype=np.float32)
    datay = np.asarray(datay, dtype=np.float32)
    return datax, datay


# prepare data:
def prepare_data(num, look_back=10, data_range=1000, fmul=10, complexity=3):
    # look_back is the window length
    X, Y = data_gen(num, data_range=data_range, fmul=fmul, complexity=complexity)
    # reshape:
    global data_min, data_max
    data_min = np.min(X)
    data_max = np.max(X)
    X = X/(data_max-data_min) + data_min
    Y = Y/(data_max - data_min) + data_min
    dataX, dataY = [], []
    for i in range(X.shape[0]):
        datax, datay = sequence_data(X[i], Y[i], look_back)
        dataX.append(datax)
        dataY.append(datay)
    dataX = np.squeeze(np.asarray(dataX))
    dataY = np.asarray(dataY)[:, :, np.newaxis]
    # but we still need to leave X
    return dataX, dataY, np.array(X[:, look_back:])


def main():
    # define model:
    # expected data format:(batch_size, time_steps, data_dim)
    look_back = 3
    dataX, dataY, X = prepare_data(1000, look_back=look_back)

    # model
    model = Sequential()
    model.add(CuDNNLSTM(32, return_sequences=True, input_shape=dataX.shape[1:]))
    model.add(CuDNNLSTM(64, return_sequences=True))
    model.add(CuDNNLSTM(32, return_sequences=True))
    model.add(Dense(1))
    # down sampling the dimension
    model.compile(loss='mse', optimizer='adagrad', metrics=['mae'])

    # start the training process
    # model.load_weights('model.h5')
    model.fit(dataX, dataY, batch_size=30, nb_epoch=100, verbose=2, validation_split=0.2)
    model.save_weights('model.h5')
    # predict: we only used on sample:
    p_dataX = dataX[0:1]
    predict_Y = model.predict(p_dataX)
    # plot and comparision
    plt.plot(np.squeeze(dataY[0:1]), label='Actual Y')
    plt.plot(np.squeeze(predict_Y), label='Predict Y')
    plt.plot(np.squeeze(X[0:1]), label='Source')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()