
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout ,Activation,Lambda
from keras.layers import merge
from capsulelayers2 import CapsuleLayer, PrimaryCap, Length, Mask
from tensorflow.keras.callbacks import EarlyStopping
K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings, batch_size):
    
    x = layers.Input(shape=input_shape, batch_size=batch_size)
    # print(x.shape)
    getindicelayer1 = Lambda(lambda x: x[:, :, :20])
    x1 = getindicelayer1(x)
    # print(x1.shape)
    getindicelayer2 = Lambda(lambda x: x[:, :, 20:])
    x2 = getindicelayer2(x)
    # print(x2.shape)
    A = layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='SAME', kernel_initializer='he_normal',
                      activation='relu', name='pssm')(x1)
    A = Dropout(0.7)(A)
    B = layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='SAME', kernel_initializer='he_normal',
                      activation='relu', name='hhm')(x2)
    B = Dropout(0.7)(B)
    merges = merge.Concatenate(axis=-1)([A, B])

    # conv1 = layers.Conv1D(filters=200, kernel_size=1, strides=1, padding='valid', kernel_initializer='he_normal',
    #                        activation='relu', name='conv1')(merge)
    # conv1 = Dropout(0.7)(conv1)
    # conv2 = layers.Conv1D(filters=200, kernel_size=9, strides=1, padding='valid', kernel_initializer='he_normal',
    #                       activation='relu', name='conv2')(conv1)
    # conv2 = Dropout(0.75)(conv2)
    primarycaps = PrimaryCap(merges, dim_capsule=8, n_channels=60, kernel_size=20, kernel_initializer='he_normal',
                             strides=1, padding='valid', dropout=0.2)
    # dim_capsule_dim2 = 10

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
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
    # return tf.reduce_mean(tf.square(y_pred))
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

    return tf.reduce_mean(tf.reduce_sum(L, 1))


def train(model, data, earlystop, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data
    # print(data)

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.95 ** epoch))
    if (
            earlystop is not None):  # use early_stop to control nb_epoch there must contain a validation if not provided will select one
        early_stopping = EarlyStopping(monitor='val_capsnet_loss', patience=earlystop)
        nb_epoch = 1
    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    """
       # Training without data augmentation:
       model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
                 validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
       """

    # # Begin: Training with data augmentation ---------------------------------------------------------------------#
    # def train_generator(x, y, batch_size, shift_fraction=0.):
    #     train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
    #                                        height_shift_range=shift_fraction,
    #                                        horizontal_flip=True)  # shift up to 2 pixel for MNIST
    #     generator = train_datagen.flow(x, y, batch_size=batch_size)
    #     while 1:
    #         x_batch, y_batch = generator.next()
    #         yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit([x_train, y_train], [y_train, x_train],
              batch_size=args.batch_size,
              epochs=args.epochs,
              # steps_per_epoch=int(y_train.shape[0] // args.batch_size),
              validation_data=[[x_test, y_test], [y_test, x_test]],
              class_weight=None,
              callbacks=[early_stopping, log, tb, checkpoint, lr_decay])

    '''
    # model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#
    '''
    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    # from utils import plot_log
    # plot_log(args.save_dir + '/log.csv', show=True)

    return model



def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-' * 30 + 'Begin: test' + '-' * 30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

    # img = combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    # image = img * 255
    # Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    # print()
    # print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    # print('-' * 30 + 'End: test' + '-' * 30)
    # plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    # plt.show()


# def manipulate_latent(model, data, args):
#     print('-' * 30 + 'Begin: manipulate' + '-' * 30)
#     x_test, y_test = data
#     index = np.argmax(y_test, 1) == args.digit
#     number = np.random.randint(low=0, high=sum(index) - 1)
#     x, y = x_test[index][number], y_test[index][number]
#     x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
#     noise = np.zeros([1, 10, 16])
#     x_recons = []
#     for dim in range(16):
#         for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
#             tmp = np.copy(noise)
#             tmp[:, :, dim] = r
#             x_recon = model.predict([x, y, tmp])
#             x_recons.append(x_recon)
#
#     x_recons = np.concatenate(x_recons)
#
#     img = combine_images(x_recons, height=16)
#     image = img * 255
#     Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
#     print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
#     print('-' * 30 + 'End: manipulate' + '-' * 30)





if __name__ == "__main__":
    import os
    import argparse
    # from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import callbacks
    tf.compat.v1.disable_eager_execution()
    gpu_id = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.system('echo $CUDA_VISIBLE_DEVICES')
    import keras
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth = True  # TensorFlow按需分配显存
    config.gpu_options.per_process_gpu_memory_fraction = 0.4  # 指定显存分配比例
    tf_session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
    tf.compat.v1.keras.backend.set_session(tf_session)

    # setting the hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--lam_recon', default=0.392, type=float)  # 784 * 0.0005, paper uses sum of SE, here uses MSE
    parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
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
    i = '01'
    train1 = np.load('data/PSSMtrain_81.npz')
    test1 = np.load('data/PSSMtest_81.npz')
    train2 = np.load('data/HHtrain_81.npz')
    test2 = np.load('data/HHtest_81.npz')

    x_train1 = train1['pssm']  # pssm
    y_train1 = train1['frag']
    x_train2 = train2['hhm']
    y_train2 = train2['frag']

    x_test1 = test1['pssm']  # pssm
    y_test1 = test1['frag']
    x_test2 = test2['hhm']
    y_test2 = test2['frag']

    if len(x_train1.shape) > 3:
        x_train1.shape = (x_train1.shape[0], x_train1.shape[2], x_train1.shape[3])

    if len(x_train2.shape) > 3:
        x_train2.shape = (x_train2.shape[0], x_train2.shape[2], x_train2.shape[3])

    if (x_test1 is not None):
        # print( x_test.shape)
        if len(x_test1.shape) > 3:
            x_test1.shape = (x_test1.shape[0], x_test1.shape[2], x_test1.shape[3])
    if (x_test2 is not None):
        # print( x_test.shape)
        if len(x_test2.shape) > 3:
            x_test2.shape = (x_test2.shape[0], x_test2.shape[2], x_test2.shape[3])

    x_train = np.concatenate((x_train1[:, :, :-1], x_train2), axis=-1)
    x_test = np.concatenate((x_test1[:, :, :-1], x_test2), axis=-1)
    y_train = train_label = y_train1
    y_test = validation_label = y_test1
    print(type(x_train.shape[1:]))

    # define model
    model, eval_model= CapsNet(input_shape=x_train.shape[1:],
                               n_class=len(np.unique(np.argmax(y_train, 1))),
                               routings=args.num_routing,
                               batch_size=args.batch_size)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if args.is_training:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), earlystop=earlystop, args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        # manipulate_latent(manipulate_model, (x_test, y_test), args)
        test(model=eval_model, data=(x_test, y_test), args=args)
