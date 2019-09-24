import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score


class model():
    def __init__(self, num_classes):
        self.chars_num = 72
        self.num_classes = num_classes  # 类别
        self.dropout = 0.5
        self.create_placeholder()
        self.model()

    def create_placeholder(self):
        # 最多字符数=1014，不够补0，否则截断
        self.input = tf.placeholder(shape=[None, 1014], dtype=tf.int32, name="input")
        self.labels = tf.placeholder(shape=[None], dtype=tf.int32, name="labels")
        self.is_training = tf.placeholder(shape=[], dtype=tf.bool, name="is_training")

    def conv_block(self, input, filters, kernel_size, is_max_pooling):
        c = tf.layers.conv1d(inputs=input, filters=filters,
                             kernel_size=kernel_size,
                             padding="valid")
        if is_max_pooling:
            c = tf.layers.max_pooling1d(inputs=c,
                                        pool_size=3,
                                        strides=3,
                                        padding="valid")
        o = tf.nn.relu(c)
        return o

    def model(self):
        input = tf.one_hot(self.input, self.chars_num)  # batch_size*1014*72
        o1 = self.conv_block(input, filters=256, kernel_size=7, is_max_pooling=True)
        o2 = self.conv_block(o1, filters=256, kernel_size=7, is_max_pooling=True)
        o3 = self.conv_block(o2, filters=256, kernel_size=3, is_max_pooling=False)
        o4 = self.conv_block(o3, filters=256, kernel_size=3, is_max_pooling=False)
        o5 = self.conv_block(o4, filters=256, kernel_size=3, is_max_pooling=False)
        o6 = self.conv_block(o5, filters=256, kernel_size=3, is_max_pooling=True)
        mlp_input = tf.layers.flatten(o6)
        mlp_1 = tf.layers.dense(inputs=mlp_input, units=1024, activation="relu")
        mlp_1 = tf.layers.dropout(inputs=mlp_1, rate=self.dropout, training=self.is_training)
        mlp_2 = tf.layers.dense(inputs=mlp_1, units=1024, activation="relu")
        mlp_2 = tf.layers.dropout(inputs=mlp_2, rate=self.dropout, training=self.is_training)

        output = tf.layers.dense(inputs=mlp_2, units=self.num_classes)
        labels = tf.one_hot(self.labels, depth=self.num_classes)
        self.loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=output))
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss_op)
        self.predict = tf.argmax(output, axis=1)

    def train(self, sess, datas, labels, batch_size):
        index = 0
        while index < len(datas):
            data_batch = datas[index:index + batch_size]
            label_batch = labels[index:index + batch_size]
            loss, _ = sess.run([self.loss_op, self.train_op], feed_dict={self.input: data_batch,
                                                                         self.labels: label_batch,
                                                                         self.is_training: True})
            if index % (batch_size * 1000) == 0:
                print("Training Loss is:", loss)
            index += batch_size

    def test(self, sess, datas, labels, batch_size):
        index = 0
        resuts = []
        while index < len(datas):
            data_batch = datas[index:index + batch_size]
            pred = sess.run(self.predict, feed_dict={self.input: data_batch, self.is_training: False})
            resuts += list(pred)
            index += batch_size
        acc = accuracy_score(labels, resuts)
        return acc


if __name__ == "__main__":
    model = model()
