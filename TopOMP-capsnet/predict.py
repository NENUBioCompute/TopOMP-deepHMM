
from keras import layers, models, optimizers
from keras import backend as K
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.layers import Dropout ,Activation,Lambda
from keras.layers.merge import Concatenate
import numpy as np
K.set_image_data_format('channels_last')
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef

#state and label
mdel=open("./hmm_parameter/mdel.txt","r")
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

def viterbi(A, Pi, Obser, state,count,y_pred):
    row, col = len(Obser), 60
    res = np.zeros((row, col))
    res2 = np.zeros_like(res)
    A, Pi = np.array(A),  np.array(Pi)
    for w in range(0,col):
        res[0,w]=y_pred[count][0][psym_dict[pstate[w]]]* Pi[w]
    #res[0, :] =  B.T[0] * Pi #把第一个观测节点对应的各状态值计算出来
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


def CapsNet(input_shape, n_class, num_routing):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)
    # print(x.shape)
    getindicelayer1 = Lambda(lambda x: x[:,:,:20])
    x1 =getindicelayer1(x)
    # print(x1.shape)
    getindicelayer2 = Lambda(lambda x: x[:,:,20:])
    x2 = getindicelayer2(x)
    # print(x2.shape)
    A = layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='SAME', kernel_initializer='he_normal',
                          activation='relu', name='pssm')(x1)
    A = Dropout(0.7)(A)
    B = layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='SAME', kernel_initializer='he_normal',
                          activation='relu', name='hhm')(x2)
    B = Dropout(0.7)(B)
    merge = Concatenate(axis=-1)([A, B])

    # conv1 = layers.Conv1D(filters=200, kernel_size=1, strides=1, padding='valid', kernel_initializer='he_normal',
    #                        activation='relu', name='conv1')(merge)
    # conv1 = Dropout(0.7)(conv1)
    # conv2 = layers.Conv1D(filters=200, kernel_size=9, strides=1, padding='valid', kernel_initializer='he_normal',
    #                       activation='relu', name='conv2')(conv1)
    # conv2 = Dropout(0.75)(conv2)
    primarycaps = PrimaryCap(merge, dim_capsule=8, n_channels=60, kernel_size=20, kernel_initializer='he_normal',
                             strides=1, padding='valid', dropout=0.2)
    # dim_capsule_dim2 = 10

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, num_routing=num_routing,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=(input_shape), name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])
    return train_model, eval_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def pred(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=args.batch_size)

    # print(y_pred,y_test)
    # np.savetxt('predlabel', y_pred)
    a = np.argmax(y_pred, 1)
    b = np.argmax(y_test, 1)
    a.tolist()
    print(a)


    ssConvertMap = {0: 'I', 1: 'O', 2: 'M'}
    result = []
    for x in range(len(a)):
        result.append(ssConvertMap[a[x]])
    result1 = ''.join(result)
    #return ''.join(result)
    # print(result1+'\n\n\n')
    # fw = open("result.txt","w")
    # fw.write(result1)
    # fw.close()

    print('-' * 50)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])
    print("Q3accuracy_score:", accuracy_score(b, a))
    print("Q3mcc_score:", matthews_corrcoef(b, a))

    for i in range(len(b)):
        if b[i]==1:
            b[i]=0
    for i in range(len(a)):
        if a[i]==1:
            a[i]=0
    print("Q2accuracy_score:", accuracy_score(b,a))
    print("Q2mcc_score:", matthews_corrcoef(b, a))


if __name__ == "__main__":

    import os
    import tensorflow as tf

    import os
    import tensorflow as tf

    gpu_id = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.system('echo $CUDA_VISIBLE_DEVICES')

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf.Session(config=tf_config)

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--lam_recon', default=0.392, type=float)  # 784 * 0.0005, paper uses sum of SE, here uses MSE
    parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('-input', dest='inputfile', type=str
                        ,
                        help='training data in fasta format. Sites followed by "#" are positive sites for a specific PTM prediction.'
                        , required=False)
    parser.add_argument('-valinput', dest='valfile', type=str
                        ,
                        help='validation data in fasta format if any. It will randomly select 10 percent of samples from the training data as a validation data set, if no validation file is provided.'
                        , required=False, default=None)
    parser.add_argument('-earlystop', dest='earlystop', type=int,
                        help='after the \'earlystop\' number of epochs with no improvement the training will be stopped for one bootstrap step. [Default: 20]',
                        required=False, default=20)
    args = parser.parse_args()
    print(args)
    inputfile = args.inputfile;
    valfile = args.valfile;
    earlystop = args.earlystop;
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    import numpy as np

    # load data

    test1 = np.load('data/PSSMtest_81.npz')
    test2 = np.load('data/HHtest_81.npz')

    x_test1 = test1['pssm']  # pssm
    y_test1 = test1['frag']
    x_test2 = test2['hhm']
    y_test2 = test2['frag']



    if len(x_test1.shape) > 3:
        x_test1.shape = (x_test1.shape[0], x_test1.shape[2], x_test1.shape[3])


    if len(x_test2.shape) > 3:
        x_test2.shape = (x_test2.shape[0], x_test2.shape[2], x_test2.shape[3])


    x_test = np.concatenate((x_test1[:, :, :-1], x_test2), axis=-1)
    y_test = validation_label = y_test1


    # define model
    model, eval_model = CapsNet(input_shape=x_test.shape[1:],
                                n_class=3,
                                num_routing=args.num_routing)
    model.summary()

    # train or test

    # model.load_weights(args.weights)
    model.load_weights("./result/weights-08.h5")

    pred(model=eval_model, data=(x_test, y_test))

    # model = eval_model
    # y_pred, x_recon = model.predict(x_test, batch_size=args.batch_size)
    #
    # a = np.argmax(y_pred, 1)
    # # a.tolist()
    # # print(a)
    # ssConvertMap = {0: 'I', 1: 'O', 2: 'M'}
    # result = []
    # for x in range(len(a)):
    #     result.append(ssConvertMap[a[x]])
    # result1 = ''.join(result)
    # # return ''.join(result)
    # # print(result1 + '\n\n\n')
    #
    # fasta = './data/test.txt'
    # with open(fasta) as get_fasta:
    #     score_dataset = []
    #     temp = get_fasta.readline()
    #     pdb_id = ""
    #     index = 0
    #     while temp:
    #         if (temp[0] == ">"):
    #             pdb_id = temp[1:].strip()
    #             temp = get_fasta.readline()
    #             continue
    #         score_line = []
    #         for i in temp:
    #             if (i != '\n'):
    #                 score_line.append(result[index])
    #                 index += 1
    #         score_dataset.append(score_line)
    #         temp = get_fasta.readline()
    #     y_predscore = score_dataset
    #     print(y_predscore)
    #
    #     fw = open( "result2.txt", "a")
    #     f = open(fasta, "r")
    #     l = f.readline()
    #     obs_seq = ""
    #     count = 0
    #     while l:
    #         if l[0] == ">":
    #             fw.write(l[0:7] + "|seq_len" + str(len(obs_seq)) + "\n")
    #         if l[0] != ">" and l != "\n":
    #             obs_seq = l.strip()
    #             fw.write("seq:" + obs_seq + "\n")
    #             # # y_predscore[count]
    #             # result1 = ''.join(y_predscore[count])
    #             # count = count + 1
    #             # fresult = []
    #             fw.write("pre:")
    #             for i in y_predscore[count]:
    #                 fw.write(i)
    #             fw.write("\n")
    #             count = count + 1
    #         l = f.readline()
    # fw.close()
    # #
    # #

