# : Harvey Chang
# : chnme40cs@gmail.com
# the models can mainly be divided into two catalogs:
# the first is pre_train version:
# the second is train version:
import tensorflow as tf
import numpy as np
import core_nn
import deep_rnn
import plant
import app_funcs
from matplotlib import pyplot as plt


class Model:
    def __init__(self, typ, core_nn, config, app_func):
        '''
        :param typ: type is divided into pre/rnn
        :param m: m is using m data in U
        :param n: n is using n data in Y
        :return: return a dynamic rnn model
        '''
        self.typ = typ
        # m, n are the major structure
        self.m = config['m']
        self.n = config['n']
        # mp, np are the appended structure
        self.mp = config['mp']
        self.np = config['np']

        self.max_l = max(self.m-1, self.n, self.mp-1, self.np)
        if ((self.m-1) == self.max_l) or ((self.mp-1) == self.max_l):
            self.axis = 0
        elif (self.n == self.max_l) or (self.np == self.max_l):
            self.axis = 1
        # default is self.axis = 1
        # configs:
        self.config = config
        self.core_nn = core_nn
        self.app_func = app_func
        # train data
        self.batch_size = self.config['batch_size']
        self.time_step = 0
        self.channel = 0

        self.train_dataX = 0
        self.train_dataY = 0
        self.train_original_x = 0

        # val data
        self.val_dataX = 0
        self.val_dataY = 0
        self.val_original_x = 0

        # tensor part:
        self.X = 0
        self.Y = 0
        self.Xp = 0
        # append part:
        self.train_dataXp = 0
        self.train_dataYp = 0
        self.train_original_xp = 0

        self.val_dataXp = 0
        self.val_dataYp = 0
        self.val_original_xp = 0

    def network(self):
        if self.typ == 'pre':
            with tf.variable_scope('rnn/deep_rnn'):
                self.output = self.pre_version()
                # the name scope is used to make a same name
        elif self.typ == 'rnn':
            self.output = self.rnn_version()

        if self.config['append']:
            mid_output = self.append_version()
            self.output += mid_output
        # add loss
        if self.config['target'] == 'action':
            self.loss = tf.nn.l2_loss((self.Y - self.output))
        elif self.config['target'] == 'classification':
            self.loss = tf.nn.softmax_cross_entropy_with_logits(self.output, self.Y)
        self.opt = tf.train.AdagradOptimizer(learning_rate=self.config['learning_rate'], initial_accumulator_value=1e-6)
        self.train_op = self.opt.minimize(self.loss)

        self.param_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(self.param_list)
        # start training:
        self.init = tf.global_variables_initializer()

    def pre_version(self):
        # get the input tensor:
        self.X = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.channel])
        self.Y = tf.placeholder(tf.float32, [self.batch_size, self.time_step, 1])
        outputs = self.core_nn(self.X)
        return outputs

    def rnn_version(self):

        self.X = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.channel])
        self.Y = tf.placeholder(tf.float32, [self.batch_size, self.time_step, 1])

        cell = deep_rnn.Deep_rnn(1, self.core_nn)
        init_state = tf.constant(np.zeros([self.batch_size, self.n]), dtype=tf.float32)  # n is y dimension
        outputs, final_state = tf.nn.dynamic_rnn(cell, self.X, initial_state=init_state,
                                                 time_major=False, swap_memory=True)
        return outputs

    def append_version(self):
        # this part is the append part:
        # this part is used to fix the major part or add linear structure into it.
        self.Xp = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.mp + self.np])
        output = self.app_func(self.Xp)
        return output

    def add_data(self, X, Y, data_type='train'):
        # eat X and Y into Model
        # X shape(N, 1000)
        # Y shape(N, 1000)
        # using pre version of split:
        dataX, dataY = [], []
        dataXp = []
        for i in range(X.shape[0]):
            if self.typ == 'pre':
                if self.config['reverse'] == 'none':
                    datax, datay = plant.sequence_data_v2(X[i], Y[i], self.m, self.n, self.max_l, self.axis)
                elif (self.config['reverse'] == 'left') or (self.config['reverse'] == 'right'):
                    datax, datay = plant.sequence_data_v2(Y[i], X[i], self.m, self.n, self.max_l, self.axis)
            elif self.typ == 'rnn':
                datax, datay = plant.sequence_data(X[i], Y[i], self.m)

            dataX.append(datax)
            dataY.append(datay)

            if self.config['append']:
                if self.config['reverse'] == 'none':
                    dataxp, _ = plant.sequence_data_v2(X[i], Y[i], self.mp, self.np, self.max_l, self.axis)
                elif (self.config['reverse'] == 'left') or (self.config['reverse'] == 'right'):
                    dataxp, _ = plant.sequence_data_v2(Y[i], X[i], self.mp, self.np, self.max_l, self.axis)
                dataXp.append(dataxp)

        if data_type == 'train':
            self.train_dataX = np.asarray(dataX)
            self.train_dataY = np.asarray(dataY)[:, :, np.newaxis]
            self.train_original_x = np.array(X[:, self.m:])
            # log the shape when adding trainX:
            self.data_num = self.train_dataX.shape[0]
            self.time_step = self.train_dataX.shape[1]
            self.channel = self.train_dataX.shape[2]
            # add appended part:
            if self.config['append']:
                self.train_dataXp = np.asarray(dataXp)

        if data_type == 'validation':
            self.val_dataX = np.asarray(dataX)
            self.val_dataY = np.asarray(dataY)[:, :, np.newaxis]
            self.val_original_x = np.array(X[:, self.m:])
            if self.config['append']:
                self.val_dataXp = np.asarray(dataXp)

    def train(self):
        # begin training process:
        self.sess = tf.Session()
        self.sess.run(self.init)
        if self.config['restore']:
          if (self.config['typ'] == 'rnn' and self.config['first_rnn']):
            
            self.saver.restore(self.sess, self.config['pre_log_dir'])
          else:
            self.saver.restore(self.sess, self.config['log_dir'])
            print('model loaded!')
            # print(self.sess.run(self.param_list))
        # self.exam_result()
        step = 0
        while step < self.config['training_epochs']:
            for i in range(int((1.0 * self.data_num) / self.batch_size)):
                datax = self.train_dataX[i * self.batch_size:(i + 1) * self.batch_size]
                datay = self.train_dataY[i * self.batch_size:(i + 1) * self.batch_size]
                # dynamic data generating:
                if not self.config['append']:
                    _, r_loss, pred = self.sess.run([self.train_op, self.loss, self.output], feed_dict={
                        self.X: datax,
                        self.Y: datay
                    })
                else:
                    dataxp = self.train_dataXp[i * self.batch_size:(i + 1) * self.batch_size]
                    _, r_loss, pred = self.sess.run([self.train_op, self.loss, self.output], feed_dict={
                        self.X: datax,
                        self.Y: datay,
                        self.Xp: dataxp
                    })
                # after training one epoch: exam the validation set:
            # again:
            if not self.config['append']:
                val_loss, pred = self.sess.run([self.loss, self.output], feed_dict={
                    self.X: self.val_dataX,
                    self.Y: self.val_dataY
                })
            else:
                val_loss, pred = self.sess.run([self.loss, self.output], feed_dict={
                    self.X: self.val_dataX,
                    self.Y: self.val_dataY,
                    self.Xp: self.val_dataXp
                })
            print('training is {}|validation: {}| step: {}'.format(r_loss, val_loss, step))
            step += 1
        self.saver.save(self.sess, self.config['log_dir'])
        print('model saved')
        # print(self.sess.run(self.param_list))
        # self.exam_result()
        plt.plot(pred[0] - self.val_dataY[0], label='error')
        plt.plot(np.squeeze(self.val_dataY[0]), label='Actual Y')
        plt.plot(np.squeeze(pred[0]), label='Predict Y')
        plt.legend(loc='upper right')
        plt.show()

    def exam_result(self):
        # begin training process:
        self.sess = tf.Session()
        self.sess.run(self.init)
        if self.config['restore']:
            if (self.config['typ'] == 'rnn' and self.config['first_rnn']):

                self.saver.restore(self.sess, self.config['pre_log_dir'])
            else:
                self.saver.restore(self.sess, self.config['log_dir'])
                print('model loaded!')
                # print(self.sess.run(self.param_list))
                # self.exam_result()
        if not self.config['append']:
            val_loss, pred = self.sess.run([self.loss, self.output], feed_dict={
                self.X: self.val_dataX,
                self.Y: self.val_dataY
            })
        else:
            val_loss, pred = self.sess.run([self.loss, self.output], feed_dict={
                self.X: self.val_dataX,
                self.Y: self.val_dataY,
                self.Xp: self.val_dataXp
            })

        plt.plot(np.squeeze(self.val_dataY[0]), label='Actual Y')
        plt.plot(np.squeeze(pred[0]), label='Predict Y')
        plt.plot(pred[0] - self.val_dataY[0], label='error')
        plt.legend(loc='upper right')
        plt.show()


        # draw distribution:
        plt.subplot(211)
        for i in range(self.config['batch_size']):
            plt.scatter(self.val_dataX[i, :, -1], pred[i, :])

        plt.subplot(212)
        for i in range(self.config['batch_size']):
            plt.scatter(self.val_dataX[i, :, -1], self.val_dataY[i, :, -1])

        plt.show()

    def right_reverse_exam(self):
        # load model:
        self.sess = tf.Session()
        self.sess.run(self.init)
        if self.config['restore']:
            if (self.config['typ'] == 'rnn' and self.config['first_rnn']):

                self.saver.restore(self.sess, self.config['pre_log_dir'])
            else:
                self.saver.restore(self.sess, self.config['log_dir'])
                print('model loaded!')

        # input data:
        input_data = self.val_dataX[0:1, :, self.m-1]  # we only use one data in this problem
        for i in range(input_data.shape[1]):
            # run in the time sequence:
            pass






def main():
    my_core_nn = core_nn.nn_wrapper(1)


