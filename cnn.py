import os
import sys
import timeit
import numpy as np
import cPickle as pkl
import sklearn.metrics as metrics
import theano
import theano.tensor as T
from layers import load_data, ConvFeat, DMLP
from solvers import sgd_momentum, nesterov_momentum, adam


class CNN(object):
    def __init__(self, convnames, x, y, h, w, batch_size, nkerns, filtersizes, poolsizes, strides,
            dmlpnames, is_train, nodenums, ps, l1_reg):
        rng = np.random.RandomState(23456)
        self.x = x
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


def test_cnn(dataset='./data/ubiquitous.pkl',
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
             l1_reg=0.05,
             solver=adam,
             init_learning_rate=1e-3,
             init_momentum=.05,
             n_epochs=5000):
    ################
    # LOAD DATASET #
    ################
    print '... load dataset {}'.format(dataset)
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
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
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size],
                is_train: np.cast['int32'](0)})
    validate_model = theano.function(
            inputs=[index],
            outputs=[classifier.errors, classifier.p_y_given_x],
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size],
                is_train: np.cast['int32'](0)})
    learning_rate = theano.shared(
            value=np.cast[theano.config.floatX](init_learning_rate),
            name='learning_rate')
    momentum = theano.shared(
            value=np.cast[theano.config.floatX](init_momentum),
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
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size],
                is_train: np.cast['int32'](1)})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    patience = 1000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience/2)
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()
    epoch = 0
    done_looping = False

    y_true = test_set_y[:batch_size * n_test_batches]
    y_true = y_true.eval()
    N = y_true.max()

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        print "momentum:\t{}".format(momentum.get_value())
        print "learning_rate:\t{}".format(learning_rate.get_value())
        for batch_idx in xrange(n_train_batches):
            iternum = (epoch - 1) * n_train_batches + batch_idx
            if iternum % 100 == 0: print 'training @ iternum = ', iternum
            # train minibatch
            _ = train_model(batch_idx)
            # validate model
            if (iternum + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i)[0] for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('[Valid]\tepoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, batch_idx + 1, n_train_batches, this_validation_loss * 100.))
                # check if best model
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iternum * patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iternum
                    # save the best model
                    with open('cnn.model', 'wb') as f: pkl.dump(classifier, f)
                    # test model
                    test_losses = [test_model(i)[0] for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print(('[Test]\tepoch %i, minibatch %i/%i, test error %f %%') %
                          (epoch, batch_idx + 1, n_train_batches, test_score * 100.))
                    if N == 1:
                        test_py = [test_model(i)[1] for i in xrange(n_test_batches)]
                        y_scores = np.array([a[1] for batch in test_py for a in batch])
                        y_pred = np.array([score>0.5 for score in list(y_scores)])
                        fpr, tpr, _ = metrics.roc_curve(y_true, y_scores)
                        precision, recall, _ = metrics.precision_recall_curve(y_true, y_scores)
                        print '[Test]\tROC AUC score is {}'.format(metrics.roc_auc_score(y_true, y_scores))
                        print '[Test]\tPrecision score is {}'.format(metrics.precision_score(y_true, y_pred))

            if patience <= iternum:
                done_looping = True
                break
        if momentum.get_value() < .9:
            new_momentum = 1. - (1. - momentum.get_value()) * 0.98
            momentum.set_value(np.cast[theano.config.floatX](new_momentum))
        new_learning_rate = learning_rate.get_value() * 0.99
        learning_rate.set_value(np.cast[theano.config.floatX](new_learning_rate))

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ######################
    # TEST LEARNED MODEL #
    ######################
    classifier = pkl.load(open('cnn.model', 'r'))
    predictmodel = theano.function(
            inputs=[classifier.x],
            outputs=[classifier.y_pred],
            givens={
                classifier.is_train: np.cast['int32'](0)})
    test_set_x = test_set_x.get_value()
    y_pred = predictmodel(test_set_x)
    y_true = test_set_y.get_value()
    print y_pred
    print y_true


if __name__ == '__main__':
    test_cnn(dataset='../DeepLearningTutorials/data/mnist.pkl.gz',
             h=28,
             w=28,
             batch_size=400,
             convnames=['conv0', 'conv1'],
             nkerns=[100, 20],
             filtersizes=[(4,4), (4,4)],
             poolsizes=[(2,2), (2,2)],
             strides=[(2,2), (2,2)],
             nodenums=[100, 20, 10],
             dmlpnames=['fc0', 'fc1', 'softmax'],
             ps=[0.7, 0.7],
             solver=nesterov_momentum,
             init_learning_rate=.1,
             init_momentum=.05,
             n_epochs=100)
