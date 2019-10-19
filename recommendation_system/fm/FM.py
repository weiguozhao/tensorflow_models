import argparse
import logging

from sklearn.metrics import *

from util import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class FM(object):
    def __init__(self, num_classes, k, lr, batch_size, feature_length, reg_l1, reg_l2):
        self.num_classes = num_classes
        self.k = k  # 隐因子的维度
        self.lr = lr  # 学习率
        self.batch_size = batch_size
        self.p = feature_length  # 特征数量
        self.reg_l1 = reg_l1  # l1正则系数
        self.reg_l2 = reg_l2  # l2正则系数

    def add_input(self):
        with tf.name_scope("input_layer"):
            self.X = tf.placeholder('float32', [None, self.p])  # 输入 n * feature_num
            self.y = tf.placeholder('float32', [None, self.num_classes])  # one-hot format
            self.keep_prob = tf.placeholder('float32')  # 带有dropout

    def inference(self):
        """
        y'(x) = w0 + sum( wi * xi ) + 0.5 * sum( (vi xi)**2 - vi**2 * xi**2 )
        """
        with tf.variable_scope('linear_layer'):
            # 单独的全局bias
            w0 = tf.get_variable(name='w0',
                                 shape=[self.num_classes],
                                 initializer=tf.zeros_initializer())
            # 线性乘积部分
            self.w = tf.get_variable(name='w',
                                     shape=[self.p, num_classes],
                                     initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # [n, feature_num] * [feature_num, num_classes] -> [n, num_classes]
            # [n, num_classes] + [feature_num] -> [n, num_classes]
            self.linear_terms = tf.add(tf.matmul(self.X, self.w), w0)

        with tf.variable_scope('interaction_layer'):
            # 特征交叉部分
            self.v = tf.get_variable(name='v',
                                     shape=[self.p, self.k],
                                     initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # []
            self.interaction_terms = tf.multiply(0.5,
                                                 tf.reduce_mean(tf.subtract(tf.pow(tf.matmul(self.X, self.v), 2),
                                                                            tf.matmul(self.X, tf.pow(self.v, 2))),
                                                                1, keep_dims=True))
        with tf.name_scope("predict_layer"):
            self.y_out = tf.add(self.linear_terms, self.interaction_terms)
            if self.num_classes == 2:
                self.y_out_prob = tf.nn.sigmoid(self.y_out)
            elif self.num_classes > 2:
                self.y_out_prob = tf.nn.softmax(self.y_out)

    def add_loss(self):
        with tf.name_scope("loss_layer"):
            if self.num_classes == 2:
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
            elif self.num_classes > 2:
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
            mean_loss = tf.reduce_mean(cross_entropy)
            self.loss = mean_loss
            tf.summary.scalar('loss', self.loss)

    def add_accuracy(self):
        with tf.name_scope("accuracy_layer"):
            # accuracy
            self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_out, 1), tf.float32),
                                               tf.cast(tf.argmax(self.y, 1), tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            # add summary to accuracy
            tf.summary.scalar('accuracy', self.accuracy)

    def train(self):
        self.global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.FtrlOptimizer(self.lr,
                                           l1_regularization_strength=self.reg_l1,
                                           l2_regularization_strength=self.reg_l2)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        self.add_input()
        self.inference()
        self.add_loss()
        self.add_accuracy()
        self.train()


def train_model(sess, model, epochs=100, print_every=50):
    """training model"""
    # Merge all the summaries and write them out to train_logs
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('logs', sess.graph)

    # get number of batches
    num_batches = len(x_train) // batch_size + 1

    for e in range(epochs):
        num_samples = 0
        losses = []
        for ibatch in range(num_batches):
            # batch_size data
            batch_x, batch_y = next(batch_gen)
            batch_y = np.array(batch_y).astype(np.float32)
            actual_batch_size = len(batch_y)
            # create a feed dictionary for this batch
            feed_dict = {model.X: batch_x,
                         model.y: batch_y,
                         model.keep_prob: 1.0}

            loss, accuracy, summary, global_step, _ = sess.run([model.loss, model.accuracy,
                                                                merged, model.global_step,
                                                                model.train_op], feed_dict=feed_dict)
            # aggregate performance stats
            losses.append(loss * actual_batch_size)
            num_samples += actual_batch_size
            # Record summaries and train.csv-set accuracy
            train_writer.add_summary(summary, global_step=global_step)
            # print training loss and accuracy
            if global_step % print_every == 0:
                logging.info("Iteration {0}: with minibatch training loss = {1} and accuracy of {2}"
                             .format(global_step, loss, accuracy))
                saver.save(sess, "checkpoints/model", global_step=global_step)
        # print loss of one epoch
        total_loss = np.sum(losses) / num_samples
        print("Epoch {1}, Overall loss = {0:.3g}".format(total_loss, e + 1))


def test_model(sess, model, print_every=50):
    """training model"""
    # get testing data, iterable
    all_ids = []
    all_clicks = []
    # get number of batches
    num_batches = len(y_test) // batch_size + 1

    for ibatch in range(num_batches):
        # batch_size data
        batch_x, batch_y = next(test_batch_gen)
        actual_batch_size = len(batch_y)
        # create a feed dictionary for this15162 batch
        feed_dict = {model.X: batch_x,
                     model.keep_prob: 1}
        # shape of [None,2]
        y_out_prob = sess.run([model.y_out_prob], feed_dict=feed_dict)
        y_out_prob = np.array(y_out_prob[0])

        batch_clicks = np.argmax(y_out_prob, axis=1)

        batch_y = np.argmax(batch_y, axis=1)

        print(confusion_matrix(batch_y, batch_clicks))
        ibatch += 1
        if ibatch % print_every == 0:
            logging.info("Iteration {0} has finished".format(ibatch))


def shuffle_list(data):
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    if shuffle:
        data = shuffle_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state("checkpoints")
    if ckpt and ckpt.model_checkpoint_path:
        logging.info("Loading parameters for the my Factorization Machine")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        logging.info("Initializing fresh parameters for the Factorization Machine")


if __name__ == '__main__':
    '''launching TensorBoard: tensorboard --logdir=path/to/log-directory'''
    # get mode (train or test)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='train or test', type=str, default="train")
    args = parser.parse_args()
    mode = args.mode
    # length of representation
    x_train, y_train, x_test, y_test = load_dataset()
    # initialize the model
    num_classes = 2
    lr = 0.01
    batch_size = 128
    k = 40
    reg_l1 = 2e-2
    reg_l2 = 0
    feature_length = x_train.shape[1]
    # initialize FM model
    batch_gen = batch_generator([x_train, y_train], batch_size)
    test_batch_gen = batch_generator([x_test, y_test], batch_size)
    model = FM(num_classes, k, lr, batch_size, feature_length, reg_l1, reg_l2)
    # build graph for model
    model.build_graph()

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        check_restore_parameters(sess, saver)
        if mode == 'train':
            print('start training...')
            train_model(sess, model, epochs=1000, print_every=500)
        if mode == 'test':
            print('start testing...')
            test_model(sess, model)
