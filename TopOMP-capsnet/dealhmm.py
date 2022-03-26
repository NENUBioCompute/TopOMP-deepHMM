import os

import keras.utils.np_utils as kutils
import numpy as np


def read_fasta(fasta_file):
    try:
        fp = open(fasta_file)
    except IOError:
        print
        'cannot open ' + fasta_file + ', check if it exist!'
        exit()
    else:
        fp = open(fasta_file,'r')
        #lines = fp.readlines()

        fasta_dict = {}  # record seq for one id
        positive_dict = {}  # record positive positions for one id
        idlist = []  # record id list sorted
        gene_id = ""
        while True:
            line = fp.readline()

            if not line:
                break
            pass
            if line.find('>') > -1:
                if gene_id != "":
                    fasta_dict[gene_id] = seq
                    positive_dict[gene_id] = frag
                seq = ""
                frag = ""
                gene_id = line.strip('\n')  # line.split('|')[1] all in > need to be id
                seq = fp.readline().strip('\n')
                frag =fp.readline().strip('\n')
            #print(gene_id)
            #print(seq)
            #print(frag)

            fasta_dict[gene_id] = seq  # last seq need to be record
            positive_dict[gene_id] = frag
            idlist.append(gene_id)

    return fasta_dict, positive_dict, idlist

def read_hmm(hhm_file):
    f = open(hhm_file)
    line = f.readline()
    while line[0] != '#':
        line = f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    seq = []
    extras = np.zeros([0, 10])
    prob = np.zeros([0, 20])
    line = f.readline()
    while line[0:2] != '//':
        lineinfo = line.split()
        seq.append(lineinfo[0])
        probs_ = [2 ** (-float(lineinfo[i]) / 1000) if lineinfo[i] != '*' else 0. for i in range(2, 22)]
        prob = np.concatenate((prob, np.matrix(probs_)), axis=0)
        line = f.readline()
        lineinfo = line.split()
        extras_ = [2 ** (-float(lineinfo[i]) / 1000) if lineinfo[i] != '*' else 0. for i in range(0, 10)]
        extras = np.concatenate((extras, np.matrix(extras_)), axis=0)
        line = f.readline()
        assert len(line.strip()) == 0
        line = f.readline()
    # return (''.join(seq),prob,extras)
    return (seq, np.concatenate((prob, extras), axis=1))


def dealhhm(file, window):
    hhmFile = file
    window = window
    [_, hhm_vector] = read_hmm(hhmFile)
    seqLength = hhm_vector.shape[0]
    tempVec_list = []
    for i in range(seqLength):
        [_, hhm_vector] = read_hmm(hhmFile)
        tempVec = np.zeros([(window * 2) + 1, 31])
        # print(tempVec.shape)
        if i < window + 1:
            # print(i)
            tempVec[0:window - i , -1] = 1
            # print(tempVec[0:window-i].shape)
            # print(hhm_vector[0:window + i + 1, 0:30].shape)
            # print(tempVec[window - i:].shape)
            tempVec[window - i:, 0:30] = hhm_vector[0:window + i + 1, 0:30]
            hhm_vector = np.reshape(tempVec, (1, (window * 2) + 1, 31))
            # print(hhm_vector)
            tempVec_list.append(hhm_vector)
        elif i > seqLength - window:
            # print(i)
            # print(tempVec[0:seqLength - i + window, 0:30].shape)
            # print(hhm_vector[i - window:, 0:30].shape)
            tempVec[0:seqLength - i + window, 0:30] = hhm_vector[i - window:, 0:30]
            tempVec[window + seqLength - i :, -1] = 1
            hhm_vector = np.reshape(tempVec, (1, (window * 2) + 1, 31))
            # print(hhm_vector)
            tempVec_list.append(hhm_vector)
        else:
            # print(i)
            # print(hhm_vector[i-window-1:i+window, 0:30].shape)
            tempVec[0:, 0:30] = hhm_vector[i - window - 1:i + window, 0:30]
            hhm_vector = np.reshape(tempVec, (1, (window * 2) + 1, 31))
            # print(hhm_vector)
            tempVec_list.append(hhm_vector)

    return tempVec_list

def getfrag(fastafile, filepath, window):
    fastaFile = fastafile
    fasta_dict, positive_dict, idlist = read_fasta(fastaFile)
    # print(idlist)
    filepath = filepath
    list1 = os.listdir(filepath)
    pssmfinal = []
    targetArr = []
    for file in list1:
        pssmFile = filepath + "/" + file
        window = window
        name = file.split(".")[0]
        print(name)
        for id in idlist:
            if str(name) in str(id):
                # print(id)
                seq = fasta_dict[id]
                frag = positive_dict[id]
                final_seq_list = []

                for pos in range(len(seq)):
                    frag_aa = frag[pos]
                    if (frag_aa == 'I'):
                        final_seq_list.append([0])
                    elif (frag_aa == 'O'):
                        final_seq_list.append([1])
                    else:
                        final_seq_list.append([2])
                # print(final_seq_list)
                # targetList = final_seq_list[:, 0]
        # target = kutils.to_categorical(final_seq_list)
        # print(type(target))
        # print(target.shape)
        # print(target)
        # targetArr = np.append(targetArr, target,axis = 0)
        # print(type(targetArr))
        pssm = dealhhm(pssmFile, window)
        # print(type(pssm[1]))
        for i in range(len(final_seq_list)):
            result = pssm[i]

            # a = result[:, 0]
            # print(result[:, 0][:, 0], final_seq_list[i])
            # print(result.reshape(42, 43))
            # print(pssm[i].shape, result.shape)
            pssmfinal.append(result)
            targetArr.append(final_seq_list[i])

    target = kutils.to_categorical(targetArr)
    return pssmfinal, target


def main():
    # for i in range(1, 31):
    #     i = str(i)
    #     i = i.zfill(2)
    #     print(i)

        fastaFile = 'data/49.fasta'
        trainfilepath = 'data/HHtrain'
        # valfileepath = '30/' + i + '/HHval'
        window = 40

        x,y = getfrag(fastaFile, trainfilepath, window)
        a = np.array(x)
        b = np.array(y)
        print(a.shape, b.shape)
        # print(a,b)
        np.savez('HHtrain_81.npz', hhm=a, frag=b)

        # x2, y2 = getfrag(fastaFile, valfileepath, window)
        # a2 = np.array(x2)
        # b2 = np.array(y2)
        # print(a2.shape, b2.shape)
        # # print(a,b)
        # np.savez('30/' + i + '/HHval' + i + '_81.npz', hhm=a2, frag=b2)
        # if len(x.shape) > 3:
        #     x.shape = (x.shape[0], x.shape[2], x.shape[3])


if __name__ == "__main__":
    main()
