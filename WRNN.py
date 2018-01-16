# : Harvey Chang
# : chnme40cs@gmail.com
# this file is a realization of WRNN:
import numpy as np
import tensorflow as tf
from plant import prepare_data
from matplotlib import pyplot as plt


class WRNN(tf.nn.rnn_cell.RNNCell):
    ''' A realization of WRNN '''
    def __init__(self, num_units, activation=tf.nn.tanh, reuse=None):
        super(WRNN, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, input, state):
        # input and state are in same shape
        # the initial state is (10,1)
        input = tf.concat([input, state], axis=-1)

        output = self.predict_model(input)
        new_state = state[:, 1:]
        new_state = tf.concat([new_state, output], axis=-1)
        return output, new_state

    def predict_model(self, layer):
        hidden_dim = [32, 64, 32]
        pre_dim = [1, 32, 64]

        for dim, p_dim in zip(hidden_dim, pre_dim):
            layer = tf.layers.dense(layer, dim,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.keras.initializers.glorot_uniform())
        output = tf.layers.dense(layer, 1, kernel_initializer=tf.contrib.keras.initializers.glorot_uniform())

        return output


# main function:
# test on minist:
def main():
    learning_rate = 0.01
    batch_size = 30

    # data preparation
    data_range = 1000  # predict a data of range 1000
    data_num = 300
    training_epochs = 50

    look_back = 2  # using 2 previous x
    y_look_back = 3  # using 3 previous y
    timesteps = data_range - look_back  # time steps

    # data prepared:
    # define model:
    # expected data format:(batch_size, time_steps, data_dim)

    X = tf.placeholder(tf.float32, [batch_size, timesteps, look_back])
    Y = tf.placeholder(tf.float32, [batch_size, timesteps, 1])

    # cell = tf.nn.rnn_cell.BasicRNNCell(1)
    cell = WRNN(1)
    # output is (batch, time_series, num_hidden) # here is 1:
    # state for lstm cell is 2 parts:
    init_state = tf.constant(np.zeros([batch_size, y_look_back]), dtype=tf.float32)  # this look_back is Y look back
    outputs, final_state = tf.nn.dynamic_rnn(cell, X, initial_state=init_state, time_major=False, swap_memory=True)
    # define loss and optimizer:

    loss = tf.nn.l2_loss((Y - outputs))
    opt = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-6)
    train_op = opt.minimize(loss)

    param_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(param_list)
    print(param_list)
    # start training:
    init = tf.global_variables_initializer()

    # data preparation:
    dataX, dataY, original_X = prepare_data(num=data_num, data_range=data_range, look_back=look_back, noise_level=1e-2)
    val_dataX, val_dataY, val_original_X = prepare_data(num=batch_size, data_range=data_range, look_back=look_back,
                                                        noise_level=1e-2)

    with tf.Session() as sess:
        sess.run(init)
        # global step
        # restore:
        saver.restore(sess, 'train_log/pre_WRNN')
        print('model loaded!')
        step = 0
        while(step < training_epochs):
            for i in range(int((1.0*data_num)/batch_size)):
                datax = dataX[i*batch_size:(i+1)*batch_size]
                datay = dataY[i * batch_size:(i + 1) * batch_size]
                # dynamic data generating:
                _, r_loss, pred = sess.run([train_op, loss, outputs], feed_dict={
                    X: datax,
                    Y: datay
                })
                # after training one epoch: exam the validation set:
            val_loss, pred = sess.run([loss, outputs], feed_dict={
                X: val_dataX,
                Y: val_dataY
            })
            print('training is {}|validation: {}| step: {}'.format(r_loss, val_loss, step))
            step += 1
        saver.save(sess, 'train_log/WRNN')
        print('model saved')
    # see the result:
    plt.plot(pred[0] - val_dataY[0], label='error')
    plt.plot(np.squeeze(val_dataY[0]), label='Actual Y')
    plt.plot(np.squeeze(pred[0]), label='Predict Y')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()