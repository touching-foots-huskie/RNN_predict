# : Harvey Chang
# : chnme40cs@gmail.com
# this file is used to define the keras model for prediction
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import CuDNNLSTM
from keras.layers import CuDNNGRU
from plant import data_gen
from plant import prepare_data
from plant import prepare_pre_train_data
import numpy as np
from matplotlib import pyplot as plt


def main():
    # define model:
    # expected data format:(batch_size, time_steps, data_dim)
    # look_back = 10
    look_back = 2
    dataX, dataY, X = prepare_pre_train_data(200, look_back=look_back, complexity=2)

    # model
    model = Sequential()
    model.add(Dense(32, input_shape=dataX.shape[1:], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    # down sampling the dimension
    model.compile(loss='mse', optimizer='adagrad', metrics=['mae'])

    # start the training process
    # model.load_weights('dense.h5')
    model.fit(dataX, dataY, batch_size=30, nb_epoch=500, verbose=2, validation_split=0.2)
    model.save_weights('dense.h5')
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