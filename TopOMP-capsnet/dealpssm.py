import numpy as np
import os
import keras.utils.np_utils as kutils

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


def logistic(t):
    return 1.0 / (1 + np.exp(-t))

def read_pssm(pssm_file):
    f = open(pssm_file)
    line = f.readlines()
    seqLength = len(line)
    npArr = np.zeros([seqLength-9, 20])
    index = 0
    # print(seqLength)
    for i in range(3, seqLength-6):
        elements = line[i].split()
        # print(elements)
        if (len(elements) == 44 or len(elements) == 22):
            npArr[index, 0:20] = [logistic(int(x)) for x in elements[2:22]]
            index = index + 1
    # print(npArr)
    return npArr


def dealpssm(file, window):
    pssmFile = file
    window = window
    hhm_vector= read_pssm(pssmFile)
    seqLength = read_pssm(pssmFile).shape[0]
    # print(seqLength)
    temp_list = []

    for i in range(seqLength):
        temp = np.zeros([(window * 2) + 1, 21])
        hhm_vector = read_pssm(pssmFile)
        # print(temp.shape)
        if i < window + 1:
            # print(i)
            temp[0:window - i , -1] = 1
            # print(temp[0:window-i].shape)
            # print(hhm_vector[0:window + i + 1, 0:30].shape)
            # print(temp[window - i:].shape)
            temp[window - i:, 0:20] = hhm_vector[0:window + i + 1, 0:20]
            hhm_vector = np.reshape(temp, (1, (window * 2) + 1, 21))
            # print(hhm_vector)
            temp_list.append(hhm_vector)
        elif i > seqLength - window:
            # print(i)
            # print(temp[0:seqLength - i + window, 0:42].shape)
            # print(hhm_vector[i - window:, 0:42].shape)
            temp[0:seqLength - i + window, 0:20] = hhm_vector[i - window:, 0:20]
            temp[window + seqLength - i :, -1] = 1
            hhm_vector = np.reshape(temp, (1, (window * 2) + 1, 21))
            # print(hhm_vector)
            temp_list.append(hhm_vector)
        else:
            # print(i)
            # print(hhm_vector[i-window-1:i+window, 0:30].shape)
            temp[0:, 0:20] = hhm_vector[i - window - 1:i + window, 0:20]
            hhm_vector = np.reshape(temp, (1, (window * 2) + 1, 21))
            # print(hhm_vector)
            temp_list.append(hhm_vector)

    return temp_list


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
        pssm = dealpssm(pssmFile, window)
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
    return pssmfinal,target


def main():
    # for i in range(21):
    #     i = '21'
    #     print(i)
    #     i = str(i)
    #     i = i.zfill(2)

        fastaFile = 'data/49.fasta'
        trainfilepath = 'data/PSSMtrain'
        # valfileepath = ''
        window = 40

        x, y = getfrag(fastaFile, trainfilepath, window)
        a = np.array(x)
        b = np.array(y)
        print(a.shape, b.shape)
        # print(a,b)
        np.savez('data/PSSMtrain_81.npz', pssm=a, frag=b)

        # x2, y2 = getfrag(fastaFile, valfileepath, window)
        # a2 = np.array(x2)
        # b2 = np.array(y2)
        # print(a2.shape, b2.shape)
        # # print(a,b)
        # np.savez('30/' + i + '/PSSMval' + i + '_81.npz', hhm=a2, frag=b2)
        # if len(x.shape) > 3:
        #     x.shape = (x.shape[0], x.shape[2], x.shape[3])



if __name__ == "__main__":
    main()
