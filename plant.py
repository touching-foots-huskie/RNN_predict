# : Harvey Chang
# : chnme40cs@gmail.com
# this file is used to describe the non-linear plant for learning:
import numpy as np
from matplotlib import pyplot as plt
import random
import scipy.io as scio


def Plant(X, No):
    if No == 1:
        return plant1(X)
    elif No == 2:
        return plant2(X)


def plant1(X):
    # X is state, Y is performance
    # we only get X at the beginning
    Y = np.zeros(3)

    def f(x1, x2, x3, x4, x5):
        return (x1*x2*x3*x5*(x3-1.0) + x4)/(1.0 + x2**2 + x3**2)

    for i in range(3, X.shape[0]):
        Y = np.append(Y, (f(Y[i-1], Y[i-2], Y[i-3], X[i], X[i-1])))

    return np.array(Y)


def plant2(X):
    # the second version of plant:
    Y = np.zeros(2)

    def f(x1, x2, x3):
        # yn, yn-1, u
        return 0.3*x1 + 0.6*x2 + 0.6*np.sin(np.pi*x3) + 0.3*np.sin(3*np.pi*x3) + 0.1*np.sin(5*np.pi(x3))

    for i in range(2, X.shape[0]):
        Y = np.append(Y, (f(Y[i-1], Y[i-2], X[i])))
    return np.array(Y)


def signal(data_range, frequency, amplitude, tp='cos'):
    # the signal is mainly sin
    x = np.array(range(data_range))/frequency
    if tp == 'sin':
        return np.sin(x)*amplitude
    elif tp == 'cos':
        return (1-np.cos(x))*amplitude
    else:
        print('no such type!')


def one_data_gen(data_range=10000, fmul=10, complexity=3, No=1):
    # return (X, Y): Y can be written as a random combination of sin(X)
    rf = lambda: data_range/(fmul*random.randint(1, 10))
    ra = lambda: random.random()
    sig = lambda: signal(data_range, rf(), ra())
    sig_out = 0
    for i in range(complexity):
        sig_out += sig()
    return (sig_out/complexity).reshape([1, -1]), (Plant(sig_out/complexity, No)).reshape([1, -1])


def data_gen(num, data_range=25000, fmul=2, complexity=3, noise_level=0.0, No=1):
    for i in range(num):
        x,y = one_data_gen(data_range=data_range, fmul=fmul, complexity=complexity, No=No)
        # scale x and y: scale at start
        max_x = np.max(x)
        min_x = np.min(x)
        x = (x-min_x)/(max_x - min_x)
        y = (y-min_x)/(max_x - min_x)
        x += 2 * noise_level * np.random.random(size=data_range) - noise_level
        y += 2 * noise_level * np.random.random(size=data_range) - noise_level
        # add noise
        if i == 0:
            X = x.reshape([1, -1])
            Y = y.reshape([1, -1])
        else:
            X = np.concatenate([X, x], axis=0)
            Y = np.concatenate([Y, y], axis=0)
    # scale the function:

    return X, Y


def sequence_data(X, Y, look_up):
    # pass test
    datax, datay = [], []
    for j in range(Y.shape[0] - look_up):
        datax.append(np.asmatrix(X[j:j + look_up]).T)
        datay.append(Y[j + look_up])
    datax = np.asarray(datax, dtype=np.float32)
    datay = np.asarray(datay, dtype=np.float32)
    return datax, datay


def sequence_data_v2(X, Y):
    # pass test
    datax, datay = [], []
    for j in range(3, Y.shape[0]):
        datax.append(np.concatenate([X[j-1:j+1],Y[j-3:j]]))
        datay.append(Y[j])
    datax = np.asarray(datax, dtype=np.float32)  # (997, 5)
    datay = np.asarray(datay, dtype=np.float32)  # (997, 1)
    return datax, datay


# prepare data:
def prepare_data(num, look_back=10, data_range=1000, fmul=10, complexity=3, noise_level=1e-2, No=1):
    # look_back is the window length
    X, Y = data_gen(num, data_range=data_range, fmul=fmul, complexity=complexity, noise_level=1e-2, No=No)
    # add little noise
    # reshape:
    # global data_min, data_max
    dataX, dataY = [], []
    for i in range(X.shape[0]):
        datax, datay = sequence_data(X[i], Y[i], look_back)
        dataX.append(datax)
        dataY.append(datay)
    dataX = np.squeeze(np.asarray(dataX))
    dataY = np.asarray(dataY)[:, :, np.newaxis]
    # but we still need to leave X
    return dataX, dataY, np.array(X[:, look_back:])


# prepare the data for (5,1) mapping

def prepare_pre_train_data(num, look_back=10, data_range=1000, fmul=10, complexity=3, noise_level=1e-2, No=1):
    # look_back is the window length
    X, Y = data_gen(num, data_range=data_range, fmul=fmul, complexity=complexity, noise_level=noise_level, No=No)
    # reshape:
    # global data_min, data_max
    dataX, dataY = [], []
    for i in range(X.shape[0]):
        datax, datay = sequence_data_v2(X[i], Y[i])
        dataX.append(datax)
        dataY.append(datay)
    dataX = np.squeeze(np.asarray(dataX))
    dataY = np.asarray(dataY)[:, :, np.newaxis]
    # but we still need to leave X
    return dataX, dataY, np.array(X[:, 3:])


if __name__ == "__main__":
    # compress the length of data
    # X, Y = prepare_pre_train_data(10)
    matlab_dataX, matlab_dataY = data_gen(10)
    matlab_dataX = (1 - matlab_dataX) * 0.1
    # print(matlab_dataX.shape)
    # print(matlab_dataY.shape)
    scio.savemat('data', {'data': matlab_dataX})