import itertools
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_LEN = 41
EMBEDDING_DIM = 100

# choose the species
names = ['A.thaliana', 'C.elegans', 'D.melanogaster', 'E.coli', 'G.subterraneus', 'G.pickeringii', 'M.musculus']
name = names[0]

if (name == 'E.coli') or (name == 'G.subterraneus') or (name == 'G.pickeringii') or (name == 'M.musculus'):
    name_groupA = name

if (name == 'A.thaliana') or (name == 'C.elegans') or (name == 'D.melanogaster'):
    name_groupB = name

Data_dir = '../Hyb_2021/%s/' % name
# Data_dir = '../Li_2020/%s/' % name

def get_tokenizer():
    f = ['a', 'c', 'g', 't']
    c = itertools.product(f, f, f, f, f, f)
    res = []
    for i in c:
        temp = i[0]+i[1]+i[2]+i[3]+i[4]+i[5]
        res.append(temp)
    res = np.array(res)
    NB_WORDS = 4097
    tokenizer = Tokenizer(num_words=NB_WORDS)
    tokenizer.fit_on_texts(res)
    acgt_index = tokenizer.word_index
    acgt_index['null'] = 0
    return tokenizer

def sentence2word(str_set):
    word_seq = []
    for sr in str_set:
        tmp = []
        for i in range(len(sr)-5):
            if('N' in sr[i:i+6]):
                tmp.append('null')
            else:
                tmp.append(sr[i:i+6])
        word_seq.append(' '.join(tmp))
    return word_seq

def word2num(wordseq, tokenizer, MAX_LEN):
    sequences = tokenizer.texts_to_sequences(wordseq)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN)
    return numseq

def sentence2num(str_set, tokenizer, MAX_LEN):
    wordseq = sentence2word(str_set)
    numseq = word2num(wordseq, tokenizer, MAX_LEN)
    return numseq

def get_data(data):
    tokenizer = get_tokenizer()
    MAX_LEN = 41
    embedding_num = sentence2num(data, tokenizer, MAX_LEN)
    return embedding_num

def get_seqdata():
    print('Process the dataset of %s ' % name)
    x_train_P_seq = open(Data_dir + '%s_train_P.txt' % name, 'r').read().splitlines()[1::2]
    x_train_N_seq = open(Data_dir + '%s_train_N.txt' % name, 'r').read().splitlines()[1::2]
    x_test_P_seq = open(Data_dir + '%s_test_P.txt' % name, 'r').read().splitlines()[1::2]
    x_test_N_seq = open(Data_dir + '%s_test_N.txt' % name, 'r').read().splitlines()[1::2]
    return x_train_P_seq, x_train_N_seq, x_test_P_seq, x_test_N_seq

def embedding(numdata):
    embedding_matrix = np.load('embedding_matrix.npy')
    data = []
    for i in range(numdata.shape[0]):
        raw = numdata[i]
        a = []
        for j in range(numdata.shape[1]):
            index = raw[j]
            s = np.array(embedding_matrix[index])
            a = np.concatenate((a, s), axis=0)
        a = np.array(a)
        data = np.concatenate((data, a), axis=0)
    data = data.reshape(numdata.shape[0], MAX_LEN * EMBEDDING_DIM)
    return data

def get_feature_I(x_train_P_seq, x_train_N_seq, x_test_P_seq, x_test_N_seq):
    x_train_P_num = get_data(x_train_P_seq)  # 每行保存了41个数字编号(samples，41）
    x_train_N_num = get_data(x_train_N_seq)
    x_test_P_num = get_data(x_test_P_seq)
    x_test_N_num = get_data(x_test_N_seq)

    X_train_P = embedding(x_train_P_num)
    X_train_N = embedding(x_train_N_num)
    X_test_P = embedding(x_test_P_num)
    X_test_N = embedding(x_test_N_num)

    return X_train_P, X_train_N, X_test_P, X_test_N

def get_label_I(x_train_P_seq, x_train_N_seq, x_test_P_seq, x_test_N_seq):
    y_train_pos = [1] * len(x_train_P_seq)
    y_train_neg = [0] * len(x_train_N_seq)
    y_test_pos = [1] * len(x_test_P_seq)
    y_test_neg = [0] * len(x_test_N_seq)

    y_train_pos = np.array(y_train_pos).reshape(len(x_train_P_seq), 1)
    y_train_neg = np.array(y_train_neg).reshape(len(x_train_N_seq), 1)
    y_test_pos = np.array(y_test_pos).reshape(len(x_test_P_seq), 1)
    y_test_neg = np.array(y_test_neg).reshape(len(x_test_N_seq), 1)

    print("Postive/Negative samples for train : %d" % (len(x_train_P_seq)))
    print("Postive/Negative samples for test : %d" % (len(x_test_P_seq)))

    return y_train_pos, y_train_neg, y_test_pos, y_test_neg

def get_feature_II(x_train_P_seq, x_train_N_seq, x_test_P_seq, x_test_N_seq):
    x_train_P_num = get_data(x_train_P_seq)  # 每行保存了41个数字编号(samples，41）
    x_train_N_num = get_data(x_train_N_seq)
    x_test_P_num = get_data(x_test_P_seq)
    x_test_N_num = get_data(x_test_N_seq)

    X_train = np.concatenate((x_train_P_num, x_train_N_num), axis=0)
    X_test = np.concatenate((x_test_P_num, x_test_N_num), axis=0)

    return X_train, X_test

def get_label_II(x_train_P_seq, x_train_N_seq, x_test_P_seq, x_test_N_seq):
    y_train_pos = [1] * len(x_train_P_seq)
    y_train_neg = [0] * len(x_train_N_seq)
    y_test_pos = [1] * len(x_test_P_seq)
    y_test_neg = [0] * len(x_test_N_seq)

    y_train_pos = np.array(y_train_pos).reshape(len(x_train_P_seq), 1)
    y_train_neg = np.array(y_train_neg).reshape(len(x_train_N_seq), 1)
    y_test_pos = np.array(y_test_pos).reshape(len(x_test_P_seq), 1)
    y_test_neg = np.array(y_test_neg).reshape(len(x_test_N_seq), 1)

    Y_train = np.concatenate((y_train_pos, y_train_neg), axis=0)
    Y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)

    return Y_train, Y_test

x_train_P_seq, x_train_N_seq, x_test_P_seq, x_test_N_seq = get_seqdata()

# pre-Hyb_Caps
if (name == 'E.coli') or (name == 'G.subterraneus') or (name == 'G.pickeringii') or (name == 'M.musculus'):
    X_train_P, X_train_N, X_test_P, X_test_N = get_feature_I(x_train_P_seq, x_train_N_seq, x_test_P_seq, x_test_N_seq)
    y_train_pos, y_train_neg, y_test_pos, y_test_neg = get_label_I(x_train_P_seq, x_train_N_seq, x_test_P_seq, x_test_N_seq)
    np.savez(Data_dir+'%s_train.npz' % name, X_train_P=X_train_P, X_train_N=X_train_N, Y_train_P=y_train_pos, Y_train_N=y_train_neg)
    np.savez(Data_dir+'%s_test.npz' % name, X_test_P=X_test_P, X_test_N=X_test_N, Y_test_P=y_test_pos, Y_test_N=y_test_neg)

# pre-Hyb_CNN
if (name == 'A.thaliana') or (name == 'C.elegans') or (name == 'D.melanogaster'):
    X_train, X_test = get_feature_II(x_train_P_seq, x_train_N_seq, x_test_P_seq, x_test_N_seq)
    Y_train, Y_test = get_label_II(x_train_P_seq, x_train_N_seq, x_test_P_seq, x_test_N_seq)
    np.savez(Data_dir+'%s_train.npz' % name, X_train=X_train, Y_train=Y_train)
    np.savez(Data_dir+'%s_test.npz' % name, X_test=X_test, Y_test=Y_test)

