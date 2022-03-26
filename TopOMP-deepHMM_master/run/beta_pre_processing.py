import numpy as np
import os
from keras.utils import to_categorical
import math
import argparse


def format(f, n):
    if round(f) == f:
        m = len(str(f)) - 1 - n
        if f / (10 ** m) == 0.0:
            return f
        else:
            return float(int(f) / (10 ** m) * (10 ** m))
    return round(f, n - len(str(int(f)))) if len(str(f)) > n + 1 else f


dict = {'C': 0, 'D': 1, 'S': 2, 'Q': 3, 'K': 4,
        'I': 5, 'P': 6, 'T': 7, 'F': 8, 'N': 9,
        'G': 10, 'H': 11, 'L': 12, 'R': 13, 'W': 14,
        'A': 15, 'V': 16, 'E': 17, 'Y': 18, 'M': 19}

class Processor:

    def data_pre_processing(self, fasta_path, hhblits_path, window_length):
        train_fasta = open(fasta_path)
        line = train_fasta.readline()
        pdb_id = ""
        x_train = []
        while line:
            if (line[0] == ">"):
                pdb_id = line[1:].strip()
                line = train_fasta.readline()
                continue
            seq_length = len(line.strip())
            length=seq_length+window_length-1
            t = int((window_length - 1) / 2)
            # -------- add hhblits feature ----------#
            list_dir = os.getcwd()
            hhm_path = hhblits_path + pdb_id + '.hhm'
            hhm_file = os.path.join(list_dir, hhm_path)
            if (os.path.exists(hhm_file)):
                with open(hhm_file) as hhm:
                    hhm_matrix = np.zeros([length, 30], float)
                    hhm_line = hhm.readline()
                    top = t - 1
                    while (hhm_line[0] != '#'):
                        hhm_line = hhm.readline()
                    for i in range(0, 5):
                        hhm_line = hhm.readline()
                    while hhm_line:
                        if (len(hhm_line.split()) == 23):
                            each_item = hhm_line.split()[2:22]
                            for idx, s in enumerate(each_item):
                                if (s == '*'):
                                    each_item[idx] = '99999'
                            for j in range(0, 20):
                                if (top == length - 1 - t):
                                    break
                                try:
                                    hhm_matrix[top, j] = 10 / (1 + math.exp(-1 * int(each_item[j]) / 2000))
                                except IndexError:
                                    pass
                        elif (len(hhm_line.split()) == 10):
                            each_item = hhm_line.split()[0:10]
                            for idx, s in enumerate(each_item):
                                if (s == '*'):
                                    each_item[idx] = '99999'
                            for j in range(20, 30):
                                if (top == length - 1 - t):
                                    break
                                try:
                                    hhm_matrix[top, j] = 10 / (1 + math.exp(-1 * int(each_item[j - 20]) / 2000))
                                except IndexError:
                                    pass
                            top += 1
                        hhm_line = hhm.readline()
                codeh=hhm_matrix
            else:
                codeh = np.zeros([length, 30], int)
                print(str(pdb_id) + " not found!!")
                # -------- add hhblits feature ----------#

            # -------- add noseq feature ----------#
            length = codeh.shape[0]
            t = int((window_length - 1) / 2)
            noSeq = np.zeros([length - window_length + 1, 1], int)
            noSeq = np.r_[np.ones([t, 1], int), noSeq]
            noSeq = np.r_[noSeq, np.ones([t, 1], int)]
            codeh = np.c_[codeh, noSeq]
            # -------- add noseq feature ----------#

            # -------- sliding window (window_length * feature) ---------#
            length = codeh.shape[0]
            feature = codeh.shape[1]
            top = 0
            buttom = window_length
            while (buttom <= length):
                x_train.append(codeh[top:buttom])
                top += 1
                buttom += 1
            # -------- sliding window (window_length * feature) ---------#
            line = train_fasta.readline()
        x_train = np.array(x_train)
        return x_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', default="../datasets/test.txt")
    parser.add_argument('--hhblits_path', default="../datasets/test_hmm/")
    args = parser.parse_args()

    processor = Processor()
    window_length = 15
    fasta = args.fasta
    hhblits_path = args.hhblits_path
    x_test = processor.data_pre_processing(fasta, hhblits_path, window_length)
    np.save('temp.npy', x_test)
    print(x_test.shape)

