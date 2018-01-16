# : Harvey Chang
# : chnme40cs@gmail.com
# this file is used to describe the non-linear plant for learning:
import numpy as np
from matplotlib import pyplot as plt
import random
import scipy.io as scio
import tqdm


def Plant(X, No):
    if No == 1:
        return plant1(X)
    elif No == 2:
        return plant2(X)


def m_Plant(X, No):
    if No == 1:
        return m_plant1(X)
    elif No == 2:
        return m_plant2(X)


def plant1(X):
    # X is state, Y is performance
    # we only get X at the beginning
    Y = np.zeros(3)

    def f(x1, x2, x3, x4, x5):
        return (x1*x2*x3*x5*(x3-1.0) + x4)/(1.0 + x2**2 + x3**2)

    for i in range(3, X.shape[0]):
        Y = np.append(Y, (f(Y[i-1], Y[i-2], Y[i-3], X[i], X[i-1])))

    return np.array(Y)


def m_plant1(X):
    # m is an array manipulation method:
    data_num, data_range = X.shape
    Y = np.zeros([data_num, 3])
    def f(x1, x2, x3, x4, x5):
        return (x1*x2*x3*x5*(x3-1.0) + x4)/(1.0 + x2**2 + x3**2)

    for i in tqdm.tqdm(range(3, data_range)):
        Y = np.concatenate([Y, f(Y[:, i-1], Y[:, i-2], Y[:, i-3], X[:, i], X[:, i-1])
                            .reshape([-1, 1])], axis=-1)

    return np.array(Y)


def plant2(X):
    # the second version of plant:
    Y = np.zeros(2)

    def f(x1, x2, x3):
        # yn, yn-1, u
        return 0.3*x1 + 0.6*x2 + 0.6*np.sin(np.pi*x3) + 0.3*np.sin(3*np.pi*x3) + 0.1*np.sin(5*np.pi*(x3))

    for i in range(2, X.shape[0]):
        Y = np.append(Y, (f(Y[i-1], Y[i-2], X[i])))
    return np.array(Y)


def m_plant2(X):
    # m is an array manipulation method:
    data_num, data_range = X.shape
    Y = np.zeros([data_num, 2])
    def f(x1, x2, x3):
        # yn, yn-1, u
        return 0.3*x1 + 0.6*x2 + 0.6*np.sin(np.pi*x3) + 0.3*np.sin(3*np.pi*x3) + 0.1*np.sin(5*np.pi*(x3))

    for i in tqdm.tqdm(range(2, data_range)):
        Y = np.concatenate([Y, f(Y[:, i-1], Y[:, i-2], X[:, i])
                            .reshape([-1, 1])], axis=-1)

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


def m_signal(data_num, filename):
    data = scio.loadmat(filename)
    if 'y' in data.keys():
        data = data['y']  # it only has one value
    else:
        data = data['Out']
    _, total_num = data.shape
    assert total_num >= data_num
    choose_data = data[:,:data_num]
    choose_data = np.transpose(choose_data, [1, 0])
    return choose_data


def m_data_gen(data_num, typ, data_range=10000, No=1, noise_level=0):
    X = m_signal(data_range, data_num, typ)
    Y = m_Plant(X, No)
    if noise_level != 0:
        X += 2 * noise_level * np.random.random(size=X.shape) - noise_level
        Y += 2 * noise_level * np.random.random(size=Y.shape) - noise_level
    return X, Y


def one_data_gen(data_range=10000, fmul=10, complexity=3, No=1):
    # return (X, Y): Y can be written as a random combination of sin(X)
    rf = lambda: data_range/(fmul*random.randint(1, 10))
    ra = lambda: random.random()
    sig = lambda: signal(data_range, rf(), ra())
    sig_out = 0
    for i in tqdm.tqdm(range(complexity)):
        sig_out += sig()
    return (sig_out/complexity).reshape([1, -1]), (Plant(sig_out/complexity, No)).reshape([1, -1])


def one_source_gen(data_range=10000, fmul=10, complexity=3, No=1):
    # return (X, Y): Y can be written as a random combination of sin(X)
    rf = lambda: data_range/(fmul*random.randint(1, 10))
    ra = lambda: random.random()
    sig = lambda: signal(data_range, rf(), ra())
    sig_out = 0
    for i in tqdm.tqdm(range(complexity)):
        sig_out += sig()
    return (sig_out/complexity).reshape([1, -1])


def source_gen(num, data_range=1000, fmul=10, complexity=1, noise_level=0.0, No=1):
    for i in tqdm.tqdm(range(num)):
        x = one_source_gen(data_range=data_range, fmul=fmul, complexity=complexity, No=No)
        # scale x and y: scale at start
        x += 2 * noise_level * np.random.random(size=data_range) - noise_level
        # add noise
        if i == 0:
            X = x.reshape([1, -1])
        else:
            X = np.concatenate([X, x], axis=0)
    # scale the function:

    return X 


def data_gen(num, data_range=1000, fmul=10, complexity=1, noise_level=0.0, No=1):
    for i in tqdm.tqdm(range(num)):
        x,y = one_data_gen(data_range=data_range, fmul=fmul, complexity=complexity, No=No)
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
        datax.append(X[j:j + look_up])
        datay.append(Y[j + look_up])
    datax = np.asarray(datax, dtype=np.float32)
    datay = np.asarray(datay, dtype=np.float32)
    return datax, datay


def sequence_data_v2(X, Y, m=2, n=3, max_l=3, axis=0):
    # m is the lookup in x
    # n is the lookup in y
    # pass test
    # n-m+1 >0 | m and n's relation should control

    datax, datay = [], []
    if axis==1:
        for j in range(max_l, Y.shape[0]):
            datax.append(np.concatenate([X[j-m+1:j+1], Y[j-n:j]]))
            datay.append(Y[j])
    elif axis==0:
        for j in range(max_l, Y.shape[0]):
            datax.append(np.concatenate([X[j-m+1:j], Y[j-n:j]]))
            datay.append(Y[j])

    datax = np.asarray(datax, dtype=np.float32)  # (997, 5)
    datay = np.asarray(datay, dtype=np.float32)  # (997, 1)
    return datax, datay


# prepare data:
def prepare_data(num, look_back=10, data_range=1000, fmul=10, complexity=3, noise_level=1e-2, No=1):
    # look_back is the window length
    X, Y = data_gen(num, data_range=data_range, fmul=fmul, complexity=complexity, noise_level=1e-2, No=No)
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
    # data = m_data_gen(100, 'prbs', No=2)
    pass

