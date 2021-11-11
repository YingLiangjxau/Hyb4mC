import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from dataprocess import name_groupA
from dataprocess import Data_dir
name = name_groupA
print('Experiment on %s dataset' % name)

class CapsNet_CirRB:
    def __init__(self):
        self.species = name
        self.threshold = 41
        self.EMBEDDING_DIM = 100
        self.n_epochs = 30
        self.batch_size = 16
        self.build_CapsNet()
        self.load_dataset()

    def squash(self, s, axis=-1, epsilon=1e-7, name=None):
        with tf.name_scope(name, default_name="squash"):
            squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=True)
            safe_norm = tf.sqrt(squared_norm + epsilon)
            squash_factor = squared_norm / (1. + squared_norm)
            unit_vector = s / safe_norm
            return squash_factor * unit_vector

    def safe_norm(self, s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
        with tf.name_scope(name, default_name="safe_norm"):
            squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)
            return tf.sqrt(squared_norm + epsilon)

    def load_dataset(self):
        print('\n****************** Get the dataset ******************\n')
        train = np.load(Data_dir + '%s_train.npz' % self.species)
        test = np.load(Data_dir + '%s_test.npz' % self.species)

        self.X_train_pos, self.X_train_neg = train['X_train_P'], train['X_train_N']
        self.Y_train_pos, self.Y_train_neg = train['Y_train_P'], train['Y_train_N']
        self.X_test_pos, self.X_test_neg = test['X_test_P'], test['X_test_N']
        self.Y_test_pos, self.Y_test_neg = test['Y_test_P'], test['Y_test_N']

        self.X_train = pd.DataFrame(np.concatenate((self.X_train_pos, self.X_train_neg), axis=0))
        y_train = np.concatenate((self.Y_train_pos, self.Y_train_neg), axis=0)
        self.Y_train = pd.Series(y_train.reshape(y_train.shape[0],))

        newindex = np.random.permutation(self.X_train.index)
        self.X_train = self.X_train.reindex(newindex)
        self.Y_train = self.Y_train.reindex(newindex)

        self.X_test = pd.DataFrame(np.concatenate((self.X_test_pos, self.X_test_neg), axis=0))
        y_test = np.concatenate((self.Y_test_pos, self.Y_test_neg), axis=0)
        self.Y_test = pd.Series(y_test.reshape(y_test.shape[0],))

    def build_CapsNet(self):

        self.X = tf.placeholder(shape=[None, self.EMBEDDING_DIM, self.threshold, 1], dtype=tf.float32, name="X")
        self.y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

        conv1_params = {
            "filters": 128,
            "kernel_size": [100, 5],
            "strides": 1,
            "padding": "valid",
            "activation": tf.nn.relu,
        }
        conv1 = tf.layers.conv2d(self.X, name="conv1", **conv1_params)
        pooling1 = tf.keras.layers.GlobalMaxPool2D()(conv1)

        # Primary Capsules
        caps1_n_maps = 16
        caps1_n_caps = caps1_n_maps * 1
        caps1_n_dims = 8

        caps1_raw = tf.reshape(pooling1, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw")
        caps1_output = self.squash(caps1_raw, name="caps1_output")

        # Digit Capsules
        caps2_n_caps = 2
        caps2_n_dims = 16
        init_sigma = 0.1

        W_init = tf.random_normal(shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims), stddev=init_sigma, dtype=tf.float32, name="W_init")
        W = tf.Variable(W_init, name="W")
        batch_size = tf.shape(self.X)[0]
        W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

        caps1_output_expanded = tf.expand_dims(caps1_output, -1, name="caps1_output_expanded")
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_tile")
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1], name="caps1_output_tiled")

        caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")#[?, 16, 2, 8, 1]

        # Dynamic routing
        raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1], dtype=np.float32, name="raw_weights")# b的 shape 为 [?, 16, 2, 1, 1]。
        routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")
        weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name="weighted_predictions")
        weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weighted_sum")

        caps2_output_round_1 = self.squash(weighted_sum, axis=-2, name="caps2_output_round_1")

        caps2_output_round_1_tiled = tf.tile(caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1], name="caps2_output_round_1_tiled")
        agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled, transpose_a=True, name="agreement")

        raw_weights_round_2 = tf.add(raw_weights, agreement, name="raw_weights_round_2")
        routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2, dim=2, name="routing_weights_round_2")
        weighted_predictions_round_2 = tf.multiply(routing_weights_round_2, caps2_predicted, name="weighted_predictions_round_2")
        weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2, axis=1, keep_dims=True, name="weighted_sum_round_2")

        caps2_output_round_2 = self.squash(weighted_sum_round_2, axis=-2, name="caps2_output_round_2")

        caps2_output = caps2_output_round_2

        y_proba = self.safe_norm(caps2_output, axis=-2, name="y_proba")
        y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
        self.y_pred_proba = tf.squeeze(y_proba, axis=[1, 3], name="y_pred_proba")
        self.y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2], name="y_pred")


        m_plus = 0.9
        m_minus = 0.1
        lambda_ = 0.5

        T = tf.one_hot(self.y, depth=caps2_n_caps, name="T")
        caps2_output_norm = self.safe_norm(caps2_output, axis=-2, keep_dims=True, name="caps2_output_norm")

        present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm), name="present_error_raw")
        present_error = tf.reshape(present_error_raw, shape=(-1, 2), name="present_error")

        absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus), name="absent_error_raw")
        absent_error = tf.reshape(absent_error_raw, shape=(-1, 2), name="absent_error")

        L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error, name="L")

        self.loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
        correct = tf.equal(self.y, self.y_pred, name="correct")
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

        optimizer = tf.train.AdamOptimizer()
        self.training_op = optimizer.minimize(self.loss, name="training_op")

    def train_CapsNet(self):
        print('\n****************** train the CapsNet ******************\n')
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        restore_checkpoint = True
        n_iterations_per_epoch = len(self.y_train) // self.batch_size
        n_iterations_validation = len(self.y_val) // self.batch_size
        best_loss_val = np.infty
        with tf.Session() as sess:
            if restore_checkpoint and tf.train.checkpoint_exists(self.checkpoint_path):
                self.saver.restore(sess, self.checkpoint_path)
            else:
                init.run()

            for epoch in range(self.n_epochs):
                for iteration in range(1, n_iterations_per_epoch + 1):
                    min_idx = (iteration - 1) * self.batch_size
                    max_idx = np.min([len(self.y_train), (iteration+1) * self.batch_size])
                    X_batch = self.x_train.iloc[min_idx:max_idx, :].values
                    y_batch = self.y_train.iloc[min_idx:max_idx].values
                    _, loss_train = sess.run(
                        [self.training_op, self.loss],
                        feed_dict={self.X: X_batch.reshape([-1, self.EMBEDDING_DIM, self.threshold, 1]),
                                   self.y: y_batch})
                    print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                        iteration, n_iterations_per_epoch,
                        iteration * 100 / n_iterations_per_epoch,
                        loss_train),
                        end="")

                loss_vals = []
                acc_vals = []
                for iteration in range(1, n_iterations_validation + 1):
                    min_idx = (iteration - 1) * self.batch_size
                    max_idx = np.min([len(self.y_val), (iteration+1) * self.batch_size])
                    X_batch = self.x_val.iloc[min_idx:max_idx, :].values
                    y_batch = self.y_val.iloc[min_idx:max_idx].values
                    loss_val, acc_val = sess.run(
                        [self.loss, self.accuracy],
                        feed_dict={self.X: X_batch.reshape([-1, self.EMBEDDING_DIM, self.threshold, 1]),
                                   self.y: y_batch})
                    loss_vals.append(loss_val)

                    acc_vals.append(acc_val)
                    print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                        iteration, n_iterations_validation,
                        iteration * 100 / n_iterations_validation),
                        end=" " * 10)
                loss_val = np.mean(loss_vals)
                acc_val = np.mean(acc_vals)
                print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
                    epoch + 1, acc_val * 100, loss_val,
                    " (improved)" if loss_val < best_loss_val else ""))

                if loss_val < best_loss_val:
                    self.saver.save(sess, self.checkpoint_path)
                    best_loss_val = loss_val

    def test_CapsNet(self):
        print('\n****************** test the CapsNet ******************\n')
        n_iterations_test = len(self.Y_test) // self.batch_size
        with tf.Session() as sess:
            self.saver.restore(sess, self.checkpoint_path)
            loss_tests = []
            acc_tests = []
            pred_proba_tests = []
            pred_value_tests = []
            y_label_tests = []
            for iteration in range(1, n_iterations_test + 1):
                min_idx = (iteration - 1) * self.batch_size
                max_idx = np.min([len(self.Y_test), iteration * self.batch_size])
                X_batch = self.X_test.iloc[min_idx:max_idx, :].values
                y_batch = self.Y_test.iloc[min_idx:max_idx].values
                loss_test, acc_test, y_pred_proba_value, y_pred_value = sess.run(
                    [self.loss, self.accuracy, self.y_pred_proba, self.y_pred],
                    feed_dict={self.X: X_batch.reshape([-1, self.EMBEDDING_DIM, self.threshold, 1]),
                               self.y: y_batch})
                loss_tests.append(loss_test)
                acc_tests.append(acc_test)
                pred_value_tests.extend(y_pred_value)
                pred_proba_tests.extend(y_pred_proba_value[:, 1])
                y_label_tests.extend(y_batch)
                print("\rTesting the model: {}/{} ({:.1f}%)".format(
                    iteration, n_iterations_test,
                    iteration * 100 / n_iterations_test),
                    end=" " * 10)
            loss_test = np.mean(loss_tests)
            acc_test = np.mean(acc_tests)
            print("\rTest accuracy: {:.4f}%  Loss: {:.6f}".format(acc_test * 100, loss_test))
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_label_tests, pred_proba_tests)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            print("\rTest AUC: {:.4f}%".format(roc_auc))
            return roc_auc, y_label_tests, pred_proba_tests, pred_value_tests

    def cross_Validation(self):
        sfolder = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
        Aucs=[]
        pred_probas = []
        pred_values = []
        y_labels = []
        for Fold, (train_idx, validation_idx) in enumerate(sfolder.split(self.X_train, self.Y_train)):
            self.x_train=self.X_train.iloc[train_idx, :]
            self.y_train=self.Y_train.iloc[train_idx]
            self.x_val=self.X_train.iloc[validation_idx, :]
            self.y_val=self.Y_train.iloc[validation_idx]
            self.checkpoint_path = "Model/"+self.species+"/model_"+str(Fold)
            # train the model
            self.train_CapsNet()
            # test the model
            roc_auc, y_label_tests, pred_proba_tests, pred_value_tests = self.test_CapsNet()
            Aucs.append(roc_auc)
            y_labels.extend(y_label_tests)
            pred_probas.extend(pred_proba_tests)
            pred_values.extend(pred_value_tests)
        print("The test AUC : %.4f " % np.mean(Aucs))
        # save the results
        print("\n********** Saving the Results **********\n")
        save_path = "./result/"+self.species+"/"
        np.savetxt(save_path+"labels.txt", y_labels)
        np.savetxt(save_path+"pred_probas.txt", pred_probas)
        print("\n********** Finished **********\n")

if __name__=="__main__":
    tf.reset_default_graph()
    np.random.seed(42)
    tf.set_random_seed(42)
    myCapsNet = CapsNet_CirRB()
    myCapsNet.cross_Validation()