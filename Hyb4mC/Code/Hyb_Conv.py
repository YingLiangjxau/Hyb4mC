import os
import keras
import numpy as np
from keras import initializers
from keras.engine.topology import Layer, InputSpec
from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from dataprocess import name_groupB
from dataprocess import Data_dir
name = name_groupB
print('Experiment on %s dataset' % name)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MAX_LEN = 41
NB_WORDS = 4097
EMBEDDING_DIM = 100
embedding_matrix = np.load('embedding_matrix.npy')

test = np.load(Data_dir + '%s_test.npz' % name)
train = np.load(Data_dir+'%s_train.npz' % name)

class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)
        if mask is not None:
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) +
                      K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def get_model():
    X = Input(shape=(MAX_LEN,))

    emb = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[
                       embedding_matrix], trainable=True)(X)

    conv_layer = Convolution1D(input_dim=100,
                               input_length=MAX_LEN,
                               nb_filter=64,
                               filter_length=20,
                               border_mode="same",
                               )
    max_pool_layer = MaxPooling1D(pool_length=int(30), stride=int(1))

    cnn = Sequential()
    cnn.add(conv_layer)
    cnn.add(Activation("relu"))
    cnn.add(max_pool_layer)
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.5))
    out = cnn(emb)

    att = AttLayer(50)(out)
    bn = BatchNormalization()(att)
    dt = Dropout(0.5)(bn)
    dt = Dense(output_dim=32, init="glorot_uniform")(dt)
    dt = Activation("relu")(dt)
    preds = Dense(1, activation='sigmoid')(dt)

    model = Model(X, preds)
    adam = keras.optimizers.adam(lr=5e-6)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model

# train the model
class roc_callback(Callback):
    def __init__(self, name):
        self.name = name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):

        self.model.save_weights(
            "./model/%s/Model%d.h5" % (self.name, epoch))

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

X_train = train['X_train']
Y_train = train['Y_train']

model = get_model()
model.summary()
print('*********** Traing %s Species specific model ***********' % name)
back = roc_callback(name=name)
history = model.fit(X_train, Y_train, epochs=89, batch_size=64, callbacks=[back])

# test the model
for epoch in [90]:
    model = get_model()
    model.load_weights("./model/%s/Model%s.h5" % (name, epoch))
    X_test = test['X_test']
    Y_test = test['Y_test']

    print('*********** Testing %s spieces specific model ***********' % name)
    y_pred = model.predict(X_test)
    auc = roc_auc_score(Y_test, y_pred)
    print("The test AUC : ", auc)
    np.savetxt('./result/%s/pred_probas.txt' % name, y_pred, fmt='%.05f')
    np.savetxt('./result/%s/labels.txt' % name, Y_test, fmt='%.05f')