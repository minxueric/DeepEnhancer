import os
import numpy as np
from random import shuffle
import hickle as hkl
import theano
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
    from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
    print 'using Lasagne.layers.dnn (faster)'
except ImportError:
    from lasagne.layers import Conv2DLayer
    from lasagne.layers import MaxPool2DLayer
    print 'using Lasagne.layers (slower)'

from lasagne.nonlinearities import softmax, rectify, leaky_rectify
from lasagne.updates import adam

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
from nolearn.lasagne import BatchIterator

from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity

floatX = theano.config.floatX

class AdjustVariable(object):
    def __init__(self, variable, target, half_life=20):
        self.variable = variable
        self.target = target
        self.half_life = half_life
    def __call__(self, nn, train_history):
        delta = self.variable.get_value() - self.target
        delta /= 2 ** (1. / self.half_life)
        self.variable.set_value(np.float32(self.target+ delta))


def main():
    ################
    # LOAD DATASET #
    ################
    dataset = './data/ubiquitous_aug.hkl'
    kfd = './data/ubiquitous_kfold.hkl'
    print('Loading dataset {}...'.format(dataset))
    X, y = hkl.load(open(dataset, 'r'))
    X = X.reshape(-1, 4, 1, 400).astype(floatX)
    y = y.astype('int32')
    print('X shape: {}, y shape: {}'.format(X.shape, y.shape))
    kf = hkl.load(open(kfd, 'r'))
    kfold = [(train, test) for train, test in kf]
    (train, test) = kfold[0]
    print('train_set size: {}, test_set size: {}'.format(len(train), len(test)))
    # shuffle +/- labels in minibatch
    print('shuffling train_set and test_set')
    shuffle(train)
    shuffle(test)
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]
    print('data prepared!')

    layers = [
            (InputLayer, {'shape': (None, 4, 1, 400)}),
            (Conv2DLayer, {'num_filters': 64, 'filter_size': (1, 4)}),
            (Conv2DLayer, {'num_filters': 64, 'filter_size': (1, 3)}),
            (Conv2DLayer, {'num_filters': 64, 'filter_size': (1, 3)}),
            (MaxPool2DLayer, {'pool_size': (1, 2)}),
            (Conv2DLayer, {'num_filters': 32, 'filter_size': (1, 2)}),
            (Conv2DLayer, {'num_filters': 32, 'filter_size': (1, 2)}),
            (Conv2DLayer, {'num_filters': 32, 'filter_size': (1, 2)}),
            (MaxPool2DLayer, {'pool_size': (1, 2)}),
            (DenseLayer, {'num_units': 64}),
            (DropoutLayer, {}),
            (DenseLayer, {'num_units': 64}),
            (DenseLayer, {'num_units': 2, 'nonlinearity': softmax})]

    net = NeuralNet(
            layers=layers,
            max_epochs=100,
            update=adam,
            update_learning_rate=1e-4,
            train_split=TrainSplit(eval_size=0.1),
            on_epoch_finished=[
                AdjustVariable(1e-4, target=0, half_life=20)],
            verbose=2)

    net.fit(X_train, y_train)
    plot_loss(net)

if __name__ == '__main__':
    main()
