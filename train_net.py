import os
import numpy as np
from random import shuffle
import hickle as hkl
import cPickle as pkl
import theano

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
    from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
    print 'Using Lasagne.layers.dnn (faster)'
except ImportError:
    from lasagne.layers import Conv2DLayer
    from lasagne.layers import MaxPool2DLayer
    print 'Using Lasagne.layers (slower)'

from lasagne.nonlinearities import softmax, rectify, leaky_rectify
from lasagne.updates import adam

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
from nolearn.lasagne import BatchIterator

from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from sklearn import metrics

floatX = theano.config.floatX

class AdjustVariable(object):
    def __init__(self, variable, target, half_life=20):
        self.variable = variable
        self.target = target
        self.half_life = half_life
    def __call__(self, nn, train_history):
        delta = self.variable.get_value() - self.target
        delta /= 2 ** (1. / self.half_life)
        self.variable.set_value(np.float32(self.target + delta))


def main(resume=None):
    l = 300
    dataset = './data/ubiquitous_train.hkl'
    print('Loading dataset {}...'.format(dataset))
    X_train, y_train = hkl.load(dataset)
    X_train = X_train.reshape(-1, 4, 1, l).astype(floatX)
    y_train = np.array(y_train, dtype='int32')
    indice = np.arange(X_train.shape[0])
    np.random.shuffle(indice)
    X_train = X_train[indice]
    y_train = y_train[indice]
    print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))

    layers = [
            (InputLayer, {'shape': (None, 4, 1, l)}),
            (Conv2DLayer, {'num_filters': 64, 'filter_size': (1, 4)}),
            (Conv2DLayer, {'num_filters': 64, 'filter_size': (1, 3)}),
            (Conv2DLayer, {'num_filters': 64, 'filter_size': (1, 3)}),
            (MaxPool2DLayer, {'pool_size': (1, 2)}),
            (Conv2DLayer, {'num_filters': 64, 'filter_size': (1, 2)}),
            (Conv2DLayer, {'num_filters': 64, 'filter_size': (1, 2)}),
            (Conv2DLayer, {'num_filters': 64, 'filter_size': (1, 2)}),
            (MaxPool2DLayer, {'pool_size': (1, 2)}),
            (DenseLayer, {'num_units': 64}),
            (DropoutLayer, {}),
            (DenseLayer, {'num_units': 64}),
            (DenseLayer, {'num_units': 2, 'nonlinearity': softmax})]

    lr = theano.shared(np.float32(1e-4))

    net = NeuralNet(
            layers=layers,
            max_epochs=100,
            update=adam,
            update_learning_rate=lr,
            train_split=TrainSplit(eval_size=0.1),
            on_epoch_finished=[
                AdjustVariable(lr, target=1e-8, half_life=20)],
            verbose=4)

    if resume != None:
        net.load_params_from(resume)

    net.fit(X_train, y_train)

    net.save_params_to('./models/net_params.pkl')

def test():
    l = 300
    dataset = './data/ubiquitous_test.hkl'
    print 'Loading dataset {}'.format(dataset)
    X_test, y_test = hkl.load(dataset)
    X_test = X_test.reshape(-1, 4, 1, l).astype(floatX)
    y_test = np.array(y_test, dtype='int32')
    print 'X_test shape: {}, y_test shape: {}'.format(X_test.shape, y_test.shape)
    test_pos_ids = hkl.load('./data/ubiquitous_test_pos_ids.hkl')
    net = pkl.load(open('./models/net.pkl', 'r'))
    y_prob = net.predict_proba(X_test)
    y_prob = y_prob[:, 1]
    f = open('./results/y_score.txt', 'w')
    f.write('id:\t predicted probability(ies)')
    y_prob_new = []
    temp = []
    for i, id, lastid in zip(
            range(len(test_pos_ids)),
            test_pos_ids,
            [None] + test_pos_ids[:-1]):
        if id != lastid:
            y_prob_new.append(temp)
            temp = []
            f.write('\n%s:\t' % id)
        temp.append(y_prob[i])
        f.write('%.4f\t' % y_prob[i])
    y_prob_new.append(temp)
    y_prob_new = [max(item) for item in y_prob_new[1:]]
    n_pos = len(y_prob_new)
    n_neg = y_test.shape[0] - len(test_pos_ids)
    y_prob_new.extend(list(y_prob[-n_neg:]))
    y_test = [1] * n_pos + [0] * n_neg

    y_pred = np.array(np.array(y_prob_new) > 0.5, dtype='int32')
    print 'ROC AUC score is {}'.format(metrics.roc_auc_score(y_test, y_prob_new))
    print 'Precision score is {}'.format(metrics.precision_score(y_test, y_pred))
    print 'Accuracy score is {}'.format(metrics.accuracy_score(y_test, y_pred))
    plot_loss(net)

if __name__ == '__main__':
    # main()
    test()
