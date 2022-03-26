#! /usr/bin/env python
# -*- coding:utf-8 -*-


from keras.models import *

from keras import backend as K

from sklearn.metrics import roc_curve, auc  ###计算roc和auc

import matplotlib.pyplot as plt
import numpy as np
K.clear_session()

K.set_image_data_format('channels_last')
np.random.seed(0)


from keras.layers import *

K.clear_session()
import numpy as np
from keras import backend as K

K.set_image_data_format('channels_last')
np.random.seed(0)
K.clear_session()
import os
import argparse
import time

#state and label
mdel=open("../hmm_parameter/mdel.txt","r")
state=mdel.readline().strip().split(" ")
ostate=mdel.readline().strip().split("   ")
pstate=mdel.readline().strip().split("   ")
osym=["I","i","M","m","n","P","w","O","o","B","E"]
psym=["I","M","O","B","E"]
osym_dict= {'I': 0,'i': 1,'M': 2,'m': 3,'n': 4,'O': 5,'o': 6}
psym_dict= {"I":0,"M":1,"O":2}

K.set_image_data_format('channels_last')
np.random.seed(0)
# one-hot map
dict_AA = {'C': 0, 'D': 1, 'S': 2, 'Q': 3, 'K': 4,
        'I': 5, 'P': 6, 'T': 7, 'F': 8, 'N': 9,
        'G': 10, 'H': 11, 'L': 12, 'R': 13, 'W': 14,
        'A': 15, 'V': 16, 'E': 17, 'Y': 18, 'M': 19}
def acu_curve(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# attention mechanism
def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(nb_time_steps, activation='relu', name='attention_dense')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

def viterbi(A, B, Pi, Obser, state,count,y_pred):
    row, col = len(Obser), 60
    res = np.zeros((row, col))
    res2 = np.zeros_like(res)
    A, B, Pi = np.array(A), np.array(B), np.array(Pi)
    res[0, :] =  B.T[0] * Pi
    for i in range(1, row):
        ob = Obser[i]
        tempres, tempres2 = [], []
        for j in range(col):
            delta = A[:, j] * res[i - 1] * y_pred[count][i][psym_dict[pstate[j]]]
            tempres2.append(np.argmax(A[:, j]*res[i - 1]))
            tempres.append(np.max(delta))
        res[i, :] = np.array(tempres)
        res2[i, :] = np.array(tempres2)

    result = []
    result.append(np.argmax(res[row-1, :]))
    i = row - 1
    while i > 0:
        result.append(res2[i][np.argmax(res[i, :])])
        i -= 1
    result.reverse()
    return res, result

def tm(l):
    startM = []
    endM = []
    for i in range(len(l) - 1):
        if l[i] != l[i + 1] and l[i + 1] == 1:
            #print(i)
            startM.append(i + 2)
        if (l[i] != l[i + 1] and l[i] == 1):
            #print(i)
            endM.append(i + 1)
    if l[len(l) - 1] == 1:
        endM.append(len(l))
    #print("startM", startM)
    #print("endM", endM)
    #print(len(endM))
    return len(endM)



if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

    '''
    cmd = python run.py --fasta ../datasets/test.txt --hhblits_path ../datasets/test_hmm/ --output_path ../result
    cmd = python run.py -f ../datasets/test.txt -p ../datasets/test_hmm/ -o ../result
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fasta',
                        default="../datasets/pdbtm_seq.fasta")
    parser.add_argument('-p', '--hhblits_path',
                        default="../datasets/hhm/")
    parser.add_argument('-o', '--output_path', default='../result')
    args = parser.parse_args()
    hhblits_path = args.hhblits_path
    fasta = args.fasta
    # calculate running time
    time_start = time.time()
    # generate data
    from beta_pre_processing import Processor
    window_length = 15
    nb_lstm_outputs = 1000
    rows, cols = window_length, 52
    nb_time_steps = window_length

    processor = Processor()
    fasta = args.fasta
    hhblits_path = args.hhblits_path
    output_path = args.output_path
    x_test = processor.data_pre_processing(fasta, hhblits_path, window_length)

    time_start = time.time()
    input = Input(shape=(window_length, 31), name='input')
    lstm_out = Bidirectional(LSTM(nb_lstm_outputs, return_sequences=True), name='bilstm1')(input)
    lstm_out = BatchNormalization(axis=-1)(lstm_out)
    lstm_out2 = Bidirectional(LSTM(nb_lstm_outputs, return_sequences=True), name='bilstm2')(lstm_out)
    attention_mul = attention_3d_block(lstm_out2)
    attention_flatten = Flatten()(attention_mul)
    fc = Dense(1024, activation='relu', kernel_initializer='random_uniform',
                bias_initializer='zeros')(attention_flatten)
    output = Dense(3, activation='softmax', name='output_1')(fc)
    model = Model(inputs=input, outputs=output)
    model.summary()

    # load weights
    model.load_weights("../lstm_train_model/trained_weights.h5")
    #nn predict score as emission
    y_pred1 = model.predict(x_test, batch_size=64)
    time_end = time.time()
    #print(y_pred1)
    Y_pred = np.argmax(y_pred1, axis=-1)
    topo_dict = {0: 'I', 1: 'M', 2: 'O'}
    with open(fasta) as get_fasta:
        score_dataset = []
        temp = get_fasta.readline()
        pdb_id = ""
        index = 0
        while temp:
            if (temp[0] == ">"):
                pdb_id = temp[1:].strip()
                temp = get_fasta.readline()
                continue
            score_line = []
            for i in temp:
                if (i != '\n'):
                    score_line.append(y_pred1[index])
                    index += 1
            score_dataset.append(score_line)
            temp = get_fasta.readline()
        y_predscore = score_dataset

    # viterbi decoding
    invisiable = {0: 'I', 1: 'M', 2: 'O'}
    invisiable_ls = [0, 1, 2]
    trainsion_probility = np.load("../hmm_parameter/A.npy")
    emission_probility = np.load("../hmm_parameter/B.npy")
    pi = np.load("../hmm_parameter/PI.npy")
    fw = open(args.output_path+"/result.txt", "a")
    f = open(fasta, "r")
    l = f.readline()
    obs_seq = ""
    count = 0
    while l:
        if l[0] == ">":
            fw.write(l[0:7] + "|seq_len" + str(len(obs_seq)) + "\n")
        if l[0] != ">" and l != "\n":
            obs_seq = l.strip()
            fw.write("seq:" + obs_seq + "\n")
            res, result = viterbi(trainsion_probility, emission_probility, pi, obs_seq, invisiable_ls, count,y_predscore)
            count = count + 1
            fresult = []
            fw.write("pre:")
            for i in result:
                fw.write(pstate[int(i)])
            fw.write("\n")
        l = f.readline()





