# : Harvey Chang
# : chnme40cs@gmail.com
# this file is used to pre_train the parameters for WRNN:
import numpy as np
import tensorflow as tf
from plant import prepare_pre_train_data
from matplotlib import pyplot as plt
from wavelet import wavelet


def predict_model(layer):

    hidden_dim = [128]
    pre_dim = [1]
    with tf.variable_scope('rnn/wrnn'):
        for dim, p_dim in zip(hidden_dim, pre_dim):
            mid_layer = tf.layers.dense(layer, dim,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.keras.initializers.glorot_uniform())
        # add wavelet:
        layer, eval_value = wavelet(mid_layer)
        output = tf.layers.dense(layer, 1, kernel_initializer=tf.contrib.keras.initializers.glorot_uniform())

    return output, eval_value


# main function:
# test on minist:
def main():
    learning_rate = 0.01
    batch_size = 30

    # data preparation
    data_range = 1000  # predict a data of range 1000
    data_num = 500
    # training_epochs = 850
    training_epochs = 1000
    look_back = 2  # using 2 previous x
    y_look_back = 3  # using 3 previous y
    timesteps = data_range - y_look_back  # time steps

    # data prepared:
    # define model:
    # expected data format:(batch_size, time_steps, data_dim)

    X = tf.placeholder(tf.float32, [batch_size, timesteps, look_back+y_look_back])
    Y = tf.placeholder(tf.float32, [batch_size, timesteps, 1])

    outputs, eval_value = predict_model(X)

    loss = tf.nn.l2_loss((Y - outputs))
    opt = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-6)
    train_op = opt.minimize(loss)

    # saver define:
    # save the parameters:
    param_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(param_list)
    print(param_list)
    # start training:
    init = tf.global_variables_initializer()

    # data preparation:
    dataX, dataY, original_X = prepare_pre_train_data(num=data_num,
                                                      data_range=data_range,
                                                      look_back=look_back,
                                                      noise_level=0)
    val_dataX, val_dataY, val_original_X = prepare_pre_train_data(num=batch_size,
                                                                  data_range=data_range,
                                                                  look_back=look_back,
                                                                  noise_level=0)

    with tf.Session() as sess:
        sess.run(init)
        # global step
        step = 0
        # print('init value is {}'.format(sess.run(eval_value)))
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

        #  save:
        saver.save(sess, 'train_log/pre_WRNN_v2')

    # see the result:
    plt.plot(pred[0] - val_dataY[0], label='error')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()