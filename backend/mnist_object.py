import math
import os
import tensorflow as tf
import os.path
from datetime import datetime
from util import *
import pickle


LOGDIR = 'logs_new_model_small/'
GITHUB_URL = 'https://raw.githubusercontent.com/mamcgrath/TensorBoard-TF-Dev-Summit-Tutorial/master/'
# 

class Net:
    def __init__(self):
        if not DEBUG:
            # don't print tensorflow debug messages
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.resume = os.path.isdir(LOGDIR)
        ### MNIST EMBEDDINGS ###
        self.mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + 'data', one_hot=True)

    ## Define CNN Layers ##
    @staticmethod
    def conv_no_act(input, size_in, size_out, name="conv"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([7, 7, size_in, size_out], stddev=0.1), name="W")
            conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
            tf.summary.histogram("weights", w)
            tf.summary.histogram("activations", conv)
            return conv

    @staticmethod
    def conv_layer(input, size_in, size_out, name="conv"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([3, 3, size_in, size_out], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
            conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
            act = tf.nn.relu(conv + b)
            normed_out = tf.contrib.layers.batch_norm(act,
                                                      center=True, scale=True,
                                                      scope=name)

            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
            return normed_out

    @staticmethod
    def pooling_layer(input, name="pool"):
        with tf.name_scope(name):
            return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    @staticmethod
    def drop_out_layer(input, prob, name="drop"):
        with tf.name_scope(name):
            return tf.nn.dropout(x=input, keep_prob=prob, name=name)

    @staticmethod
    # fully connected layer
    def fc_layer(input, size_in, size_out, name="fc"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
            act = tf.nn.relu(tf.matmul(input, w) + b)
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
            return act

    def mnist_model(self, learning_rate):
        tf.reset_default_graph()
        self.phase = True  # =  tf.placeholder(tf.bool, name='phase')
        # Setup placeholders, and reshape the data
        self.x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        tf.summary.image('input', x_image, 3)
        self.y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

        conv1 = self.conv_no_act(x_image, 1, 32, "conv_1")

        pool1 = self.pooling_layer(conv1, "pool1")

        conv2 = self.conv_no_act(pool1, 32, 64, "conv_2")
        pool2 = self.pooling_layer(conv2, "pool2")

        flattened = tf.reshape(pool2, [-1, 7 * 7 * 64])
        fc1 = self.fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
        self.prob = tf.Variable(0.4, name="prob")
        dp = self.drop_out_layer(fc1, self.prob, "drop")
        embedding_input = fc1
        embedding_size = 1024
        self.logits = self.fc_layer(dp, 1024, 10, "fc2")

        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.y), name="cross_entropy")
            tf.summary.scalar("cross_entropy", cross_entropy)

        with tf.name_scope("train"):
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)

        self.summ = tf.summary.merge_all()

        self.acc = tf.summary.scalar("accuracy_full", self.accuracy)
        self.summ_eval = tf.summary.merge([self.acc], name="acc")

        self.embedding = tf.Variable(tf.zeros([10000, embedding_size]), name="test_embedding")
        self.assignment = self.embedding.assign(embedding_input)

        # enable GPU train/inference if possible
        config = tf.ConfigProto()
        debug_print("don't" if not tf.device('/gpu:0') else "" + "recognized GPU")
        with tf.device('/gpu:0'):
            config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()

        if (os.path.isfile(os.path.join(LOGDIR, "step"))):
            self.saver.restore(self.sess, tf.train.latest_checkpoint(LOGDIR))
        else:
            self.sess.run(tf.global_variables_initializer())

        self.writer = tf.summary.FileWriter(LOGDIR)
        self.writer.add_graph(self.sess.graph)

        ## define embedding setting for TensorBoard
        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = self.embedding.name
        embedding_config.sprite.image_path = 'mnist_10k_sprite.png'
        embedding_config.metadata_path = 'labels_10k.tsv'
        # Specify the width and height of a single thumbnail.
        embedding_config.sprite.single_image_dim.extend([28, 28])
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(self.writer, config)

    def train(self):
        if os.path.exists(os.path.join(LOGDIR, "step")):
            with open(os.path.join(LOGDIR, "step"), 'rb') as file:
                step = pickle.load(file)
        else:
            step = 0
        for i in range(3000 * 50):
            batch = self.mnist.train.next_batch(100)
            ## save statistics for tensorBoard (10K data test)
            if (i + step) % 5 == 0:
                self.prob.assign(0.4)
                [train_accuracy, s, results] = self.sess.run([self.accuracy, self.summ, self.logits],
                                                             feed_dict={self.x: batch[0], self.y: batch[1]})
                self.writer.add_summary(s, (i + step))
            if (i + step) % 500 == 0:
                self.prob.assign(1)
                [accuracy, assignment, s] = self.sess.run([self.accuracy, self.assignment, self.summ_eval],
                                                          feed_dict={self.x: self.mnist.test.images[:10000],
                                                                     self.y: self.mnist.test.labels[:10000]})
                self.writer.add_summary(s, (i + step))
                print(accuracy, (i + step))
                self.saver.save(self.sess, os.path.join(LOGDIR, "model.ckpt"), (i + step))

            with open(os.path.join(LOGDIR, "step"), 'wb') as file:
                pickle.dump((i + step), file)

            ## Train
            self.sess.run(self.train_step, feed_dict={self.x: batch[0], self.y: batch[1]})

    def eval(self,img):
        # prepare image:
        reshaped_img = []
        for row in img:
            for col in row:
                reshaped_img.append(col)

        self.prob.assign(1)
        [train_accuracy, logits] = self.sess.run([self.accuracy, self.logits],
                                                 feed_dict={self.x: [reshaped_img], self.y: [[0] * 10]})
        debug_print(logits)
        eval_list = logits.tolist()[0]

        eval_val = eval_list.index(max(eval_list))

        softmax = False
        if softmax:
            sum_list = (sum([math.exp(x) for x in eval_list]))
            eval_list_percentage = [(math.exp(x) / sum_list) * 100 for x in eval_list]
        else:
            sum_list = (sum(eval_list))
            eval_list_percentage = [(x / sum_list) * 100 for x in eval_list]

        index = 0
        stat_res = ""
        for num in eval_list_percentage:
            stat_res = stat_res + 'Digit' + str(index) + ': ' + str(round(num, 1)) + "%\n"
            index += 1
        debug_print(stat_res)
        debug_print("The number is: " + str(eval_val))

        return eval_val, stat_res


def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)


def train_mnist_CNN():
    # start timer training for performance measure
    start_time = datetime.now()
    debug_print("starting training performance measure:")
    # You can try adding some more learning rates
    for learning_rate in [1E-4]:
        # Include "False" as a value to try different model architectures
        for use_two_fc in [True]:
            for use_two_conv in [True]:
                # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
                hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
                debug_print('Starting run for %s' % hparam)

                # Actually run with the new settings
                net = Net()
                net.mnist_model(learning_rate)
                net.train()

    end_time = datetime.now()
    debug_print('Training Duration: {}'.format(end_time - start_time))

# use this evaluate if you don't have Net instance
# notice that it's building the net from zero - may take several seconds
# prefer using the "eval()" method of the Net class
def eval(img):
    net = Net()
    net.mnist_model(0)

    # evaluate for the receiving img
    return net.eval(img)


# use main for CNN training only
if __name__ == '__main__':
    train_mnist_CNN()