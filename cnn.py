import os
import sys
import timeit
import numpy as np
import cPickle as pkl
import sklearn.metrics as metrics
import theano
import theano.tensor as T
from layers import load_data, ConvFeat, DMLP
from solvers import sgd, sgd_momentum, nesterov_momentum, adagrad, rmsprop, adadelta, adam


def test_cnn(init_learning_rate=.04,
             init_momentum=.05,
             n_epochs=1000,
             dataset='./data/ubiquitous.pkl',
             h=4,
             w=400,
             batch_size=500,
             convnames=['conv0'],
             nkerns=[600],
             filtersizes=[(4,15)],
             poolsizes=[(1,20)],
             strides=[(1,10)],
             nodenums=[400,50,2],
             dmlpnames=['fc0','fc1','softmax'],
             ps=[0.5,0.5],
             l1_reg=0.05,
             solver=rmsprop):
    ################
    # load dataset #
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
    rng = np.random.RandomState(23455)

    extractfeat = ConvFeat(rng=rng,
                          names=convnames,
                          input=x.reshape((batch_size, 1, h, w)),
                          h=h,
                          w=w,
                          batch_size=batch_size,
                          nkerns=nkerns,
                          filtersizes=filtersizes,
                          poolsizes=poolsizes,
                          strides=strides)
    input = extractfeat.output.flatten(2)
    nodenums = [np.prod(extractfeat.outdim[1:])] + nodenums
    classifier = DMLP(rng=rng,
                      names=dmlpnames,
                      is_train=is_train,
                      input=input,
                      nodenums=nodenums,
                      ps=ps)
    # add sparsity bias on convolution filters
    cost = classifier.negative_log_likelihood(y) + l1_reg * extractfeat.L1
    # test, validate, train functions
    test_model = theano.function(
            inputs=[index],
            outputs=[classifier.errors(y), classifier.p_y_given_x],
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size],
                is_train: np.cast['int32'](0)})
    validate_model = theano.function(
            inputs=[index],
            outputs=[classifier.errors(y), classifier.p_y_given_x],
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size],
                is_train: np.cast['int32'](0)})
    learning_rate = theano.shared(np.cast[theano.config.floatX](init_learning_rate))
    momentum = theano.shared(np.cast[theano.config.floatX](init_momentum), name='momentum')
    params = classifier.params + extractfeat.params
    if solver in [sgd_momentum, nesterov_momentum]:
        updates = solver(cost, params, learning_rate, momentum)
    else:
        updates = solver(cost, params, learning_rate)
    # updates = []
    # for param in classifier.params + extractfeat.params:
    #     param_update = theano.shared(param.get_value()*np.cast[theano.config.floatX](0.))
    #     updates.append((param,
    #         param - learning_rate * param_update))
    #     updates.append((param_update,
    #         momentum*param_update +
    #         (np.cast[theano.config.floatX](1.)-momentum)*T.grad(cost, param)))
    train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size],
                is_train: np.cast['int32'](1)})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
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
        print "momentum: {}".format(momentum.get_value())
        print "learning_rate: {}".format(learning_rate.get_value())

        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if iter % 100 == 0: print 'training @ iter = ', iter
            # train minibatch
            cost_ij = train_model(minibatch_index)
            # validate model
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i)[0] for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('[Validation] epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
                # check if best model
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    # save the best model
                    # with open('bestmodel.pkl', 'wb') as f:
                    #     pkl.dump(classifier, f)
                    # test model
                    test_losses = [
                        test_model(i)[0]
                        for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print(('[Test] epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    if N == 1:
                        test_py = [
                                test_model(i)[1]
                                for i in xrange(n_test_batches)]
                        y_scores = np.array([a[1] for batch in test_py for a in batch])
                        y_pred = np.array([score>0.5 for score in list(y_scores)])
                        fpr, tpr, _ = metrics.roc_curve(y_true, y_scores)
                        precision, recall, _ = metrics.precision_recall_curve(y_true, y_scores)
                        print '    ROC AUC score is {}'.format(metrics.roc_auc_score(y_true, y_scores))
                        print '    Precision score is {}'.format(metrics.precision_score(y_true, y_pred))

            if patience <= iter:
                done_looping = True
                break

        if momentum.get_value() < .99:
            new_momentum = 1. - (1. - momentum.get_value()) * 0.98
            momentum.set_value(np.cast[theano.config.floatX](new_momentum))
        new_learning_rate = learning_rate.get_value() * 0.99
        learning_rate.set_value(np.cast[theano.config.floatX](new_learning_rate))

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    # classifier = pkl.load(open('bestmodel.pkl', 'r'))
    # predictmodel = theano.function(
    #         inputs=[classifier.input],
    #         outputs=[classifier.y_pred])
    # test_set_x = test_set_x.get_value()
    # y_pred = predictmodel(test_set_x)
    # y_true = test_set_y.get_value()
    # print y_pred
    # print y_true


if __name__ == '__main__':
    test_cnn(init_learning_rate=.1,
             init_momentum=.05,
             n_epochs=1000,
             dataset='../DeepLearningTutorials/data/mnist.pkl.gz',
             h=28,
             w=28,
             batch_size=100,
             convnames=['conv0', 'conv1'],
             nkerns=[100, 20],
             filtersizes=[(4,4), (4,4)],
             poolsizes=[(2,2), (2,2)],
             strides=[(2,2), (2,2)],
             nodenums=[100, 20, 10],
             dmlpnames=['fc0', 'fc1', 'softmax'],
             ps=[0.9, 0.9],
             solver=sgd_momentum)
    # test_cnn()
