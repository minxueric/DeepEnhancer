import os
import sys
import timeit
import numpy as np
from random import shuffle
import cPickle as pkl
import sklearn.metrics as metrics
import theano
import theano.tensor as T
from layers import ConvFeat, DMLP
from solvers import sgd_momentum, nesterov_momentum

floatX = theano.config.floatX

def shared(data_X, data_y, borrow=True):
    shared_X = theano.shared(
            value=np.asarray(data_X, dtype=floatX),
            borrow=borrow)
    shared_y = theano.shared(
            value=np.asarray(data_y, dtype=floatX),
            borrow=borrow)
    return shared_X, T.cast(shared_y, 'int32')


class CNN(object):
    def __init__(self, convnames, x, y, h, w, batch_size, nkerns, filtersizes, poolsizes, strides,
            dmlpnames, is_train, nodenums, ps, l1_reg):
        rng = np.random.RandomState(23456)
        self.x = x
        self.y = y
        self.is_train = is_train
        extractfeat = ConvFeat(rng=rng,
                               names=convnames,
                               x=x.reshape((batch_size, 1, h, w)),
                               h=h,
                               w=w,
                               batch_size=batch_size,
                               nkerns=nkerns,
                               filtersizes=filtersizes,
                               poolsizes=poolsizes,
                               strides=strides)
        nextinput = extractfeat.output.flatten(2)
        nodenums = [np.prod(extractfeat.outdim[1:])] + nodenums
        dmlp = DMLP(rng=rng,
                    names=dmlpnames,
                    is_train=is_train,
                    x=nextinput,
                    y=y,
                    nodenums=nodenums,
                    ps=ps)
        self.params = extractfeat.params + dmlp.params
        self.cost = dmlp.negative_log_likelihood + l1_reg * extractfeat.L1
        self.errors = dmlp.errors
        self.p_y_given_x = dmlp.p_y_given_x
        self.y_pred = dmlp.y_pred


def test_cnn(dataset='./data/ubiquitous_refine.pkl',
             kfd='./data/ubiquitous_kfold.pkl',
             h=4,
             w=400,
             batch_size=400,
             convnames=['conv0'],
             nkerns=[600],
             filtersizes=[(4, 15)],
             poolsizes=[(1, 20)],
             strides=[(1, 10)],
             nodenums=[400, 50, 2],
             dmlpnames=['fc0', 'fc1', 'softmax'],
             ps=[0.8, 0.8],
             l1_reg=0.1,
             solver=nesterov_momentum,
             init_learning_rate=.04,
             init_momentum=.05,
             n_epochs=5000):
    ################
    # LOAD DATASET #
    ################
    print 'Loading dataset {}...'.format(dataset)
    X, y = pkl.load(open(dataset, 'r'))
    print X, y
    kf = pkl.load(open(kfd, 'r'))
    kfold = [(train, test) for train, test in kf]
    (train, test) = kfold[0]
    # shuffle +/- labels in minibatch
    shuffle(train)
    shuffle(test)
    train_X, train_y = shared(X[train, :], y[train])
    test_X, test_y = shared(X[test, :], y[test])
    print '... Train_set size = {}, test_set size = {}'.format(len(train), len(test))
    n_train_batches = len(train)
    n_test_batches = len(test)
    n_train_batches /= batch_size
    n_test_batches /= batch_size


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print 'Building the CNN model...'
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    is_train = T.iscalar('is_train')

    classifier = CNN(convnames=convnames,
                     x=x, y=y, h=h, w=w,
                     batch_size=batch_size,
                     nkerns=nkerns,
                     filtersizes=filtersizes,
                     poolsizes=poolsizes,
                     strides=strides,
                     dmlpnames=dmlpnames,
                     is_train=is_train,
                     nodenums=nodenums,
                     ps=ps,
                     l1_reg=l1_reg)

    test_model = theano.function(
            inputs=[index],
            outputs=[classifier.errors, classifier.p_y_given_x],
            givens={
                x: test_X[index * batch_size: (index + 1) * batch_size],
                y: test_y[index * batch_size: (index + 1) * batch_size],
                is_train: np.cast['int32'](0)})

    learning_rate = theano.shared(
            value=np.cast[floatX](init_learning_rate),
            name='learning_rate')
    momentum = theano.shared(
            value=np.cast[floatX](init_momentum),
            name='momentum')
    if solver in [sgd_momentum, nesterov_momentum]:
        updates = solver(classifier.cost, classifier.params, learning_rate, momentum)
    else:
        updates = solver(classifier.cost, classifier.params, learning_rate)

    train_model = theano.function(
            inputs=[index],
            outputs=classifier.cost,
            updates=updates,
            givens={
                x: train_X[index * batch_size: (index + 1) * batch_size],
                y: train_y[index * batch_size: (index + 1) * batch_size],
                is_train: np.cast['int32'](1)})

    ###############
    # TRAIN MODEL #
    ###############
    print 'Training the CNN model...'
    best_loss = np.inf
    test_frequency = n_train_batches
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()
    epoch = 0

    y_true = test_y[:batch_size * n_test_batches]
    y_true = y_true.eval()
    N = y_true.max()

    while (epoch < n_epochs):
        epoch = epoch + 1
        print "momentum:\t{}".format(momentum.get_value())
        print "learning_rate:\t{}".format(learning_rate.get_value())
        for batch_idx in xrange(n_train_batches):
            iternum = (epoch - 1) * n_train_batches + batch_idx
            if iternum % 100 == 0: print 'training @ iternum = ', iternum
            # train minibatch
            _ = train_model(batch_idx)
            # test model
            if (iternum + 1) % test_frequency == 0:
                test_results = [test_model(i) for i in xrange(n_test_batches)]
                (test_losses, test_py) = zip(*test_results)
                this_loss = np.mean(test_losses)
                print(('[Test]\tepoch %i, minibatch %i/%i, test error %f %%') %
                      (epoch, batch_idx + 1, n_train_batches, this_loss * 100.))
                if this_loss < best_loss:
                    best_loss = this_loss
                    best_iter = iternum
                y_scores = np.array([a[1] for batch in test_py for a in batch])
                print '[Test]\tROC AUC score is {}'.format(metrics.roc_auc_score(y_true, y_scores))
                # y_pred = np.array([score>0.5 for score in list(y_scores)])
                # fpr, tpr, _ = metrics.roc_curve(y_true, y_scores)
                # precision, recall, _ = metrics.precision_recall_curve(y_true, y_scores)
                # print '[Test]\tPrecision score is {}'.format(metrics.precision_score(y_true, y_pred))

        if momentum.get_value() < .99:
            new_momentum = 1. - (1. - momentum.get_value()) * 0.98
            momentum.set_value(np.cast[floatX](new_momentum))
        new_learning_rate = learning_rate.get_value() * 0.99
        learning_rate.set_value(np.cast[floatX](new_learning_rate))

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best model obtained at iteration %i, with test performance %f %%' %
          (best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ######################
    # TEST LEARNED MODEL #
    ######################
    classifier = pkl.load(open('cnn.model', 'r'))
    predictmodel = theano.function(
            inputs=[x],
            outputs=[classifier.y_pred],
            givens={
                is_train: np.cast['int32'](0)})
    test_set_x = test_X.get_value()
    y_pred = predictmodel(test_set_x)
    y_true = test_y.get_value()
    print y_pred
    print y_true


if __name__ == '__main__':
    test_cnn()
