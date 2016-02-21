import os
import sys
import timeit
import gzip
import numpy as np
import cPickle as pkl
import sklearn.metrics as metrics
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams

# activations
def ReLU(x):
    return T.maximum(0.0, x)

tanh = T.tanh
sigmoid = T.nnet.sigmoid
softplus = T.nnet.softplus

def load_data(dataset):
    if dataset.split('.')[-1] == 'gz':
        f = gzip.open(dataset, 'r')
    else:
        f = open(dataset, 'r')
    train_set, valid_set, test_set = pkl.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(
                np.asarray(data_x, dtype=theano.config.floatX),
                borrow=borrow)
        shared_y = theano.shared(
                np.asarray(data_y, dtype=theano.config.floatX),
                borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x,  test_set_y  = shared_dataset(test_set)

    return [(train_set_x, train_set_y),
            (valid_set_x, valid_set_y),
            (test_set_x,  test_set_y )]


class LogisticRegression(object):
    def __init__(self, name, input, n_in, n_out):
        self.input= input
        self.name = name
        # weight matrix W (n_in, n_out)
        self.W = theano.shared(
                value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                name='W',
                borrow=True)
        # bias vector b (n_out, )
        self.b = theano.shared(
                value=np.zeros((n_out,), dtype=theano.config.floatX),
                name='b',
                borrow=True)
        # p(y|x, w, b)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # params
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                    'y should have the same shape as self.y_pred',
                    ('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class ConvPoolLayer(object):
    def __init__(self, rng, name, input, filter_shape, image_shape, poolsize, stride):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.name = name

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True,
            st=stride
        )

        self.output = ReLU(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # self.L1 = abs(self.W).sum()
        self.L1 = self.W.max(axis=0).sum()

        self.params = [self.W, self.b]


class DropoutHiddenLayer(object):
    def __init__(self, rng, name, is_train, input, n_in, n_out, W=None, b=None, activation=ReLU, p=0.5):
        """p is the probability of NOT dropping out a unit"""
        self.name = name
        self.input = input
        if W is None:
            W_values = np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6./(n_in+n_out)),
                        high=np.sqrt(6./(n_in+n_out)),
                        size=(n_in, n_out)
                        ),
                    dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            # b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            # initial b to positive values, in the linear regime of ReLU
            b_values = np.ones((n_out,), dtype=theano.config.floatX) * np.cast[theano.config.floatX](0.01)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output= T.dot(input, self.W) + self.b
        output = (
                lin_output if activation is None
                else activation(lin_output))

        def drop(input, rng=rng, p=p):
            """p is the probability of NOT dropping out a unit"""
            srng = RandomStreams(rng.randint(999999))
            mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
            return input * mask

        train_output = drop(np.cast[theano.config.floatX](1./p) * output)

        self.output = T.switch(T.neq(is_train, 0), train_output, output)

        self.params = [self.W, self.b]


class DMLP(object):
    def __init__(self, rng, names, is_train, input, nodenums, ps, activation=ReLU):
        assert len(names) == len(nodenums) - 1
        assert len(names) == len(ps) + 1
        self.layers = []
        # construct first layer: names[0]
        layer = DropoutHiddenLayer(
                rng=rng,
                name=names[0],
                is_train=is_train,
                input=input,
                n_in=nodenums[0],
                n_out=nodenums[1],
                p=ps[0])
        self.layers.append(layer)
        # construct hidden layers: names[1:-1]
        if len(ps) > 1:
            for i in xrange(len(ps)-1):
                layer = DropoutHiddenLayer(
                        rng=rng,
                        name=names[i+1],
                        is_train=is_train,
                        input=self.layers[-1].output,
                        n_in=nodenums[i+1],
                        n_out=nodenums[i+2],
                        p=ps[i+1])
                self.layers.append(layer)
        # construct output layer
        layer = LogisticRegression(
                name=names[-1],
                input=self.layers[-1].output,
                n_in=nodenums[-2],
                n_out=nodenums[-1])
        self.layers.append(layer)

        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors
        self.p_y_given_x = self.layers[-1].p_y_given_x
        self.y_pred = self.layers[-1].y_pred

        self.params = [param for layer in self.layers for param in layer.params]


class ConvFeat(object):
    def __init__(self, rng, names, input, h, w, batch_size, nkerns, filtersizes, poolsizes, strides):
        self.layers = []
        # construct first layer: names[0]
        filter_shape = (nkerns[0], 1, filtersizes[0][0], filtersizes[0][1])
        image_shape = (batch_size, 1, h, w)
        poolsize = poolsizes[0]
        stride = strides[0]
        layer = ConvPoolLayer(rng=rng,
                              name=names[0],
                              input=input,
                              filter_shape=filter_shape,
                              image_shape=image_shape,
                              poolsize=poolsize,
                              stride=stride)
        self.layers.append(layer)
        h = (h - filter_shape[2] + 1 - poolsize[0]) / stride[0] + 1
        w = (w - filter_shape[3] + 1 - poolsize[1]) / stride[1] + 1
        # construct rest layers: names[1:]
        if len(names) > 1:
            for i in xrange(len(names)-1):
                filter_shape = (nkerns[i+1], nkerns[i], filtersizes[i+1][0], filtersizes[i+1][1])
                image_shape = (batch_size, nkerns[i], h, w)
                poolsize = poolsizes[i+1]
                stride = strides[i+1]
                layer = ConvPoolLayer(rng=rng,
                                      name=names[i+1],
                                      input=self.layers[-1].output,
                                      filter_shape=filter_shape,
                                      image_shape=image_shape,
                                      poolsize=poolsize,
                                      stride=stride)
                self.layers.append(layer)
                h = (h - filter_shape[2] + 1 - poolsize[0]) / stride[0] + 1
                w = (w - filter_shape[3] + 1 - poolsize[1]) / stride[1] + 1

        self.output = self.layers[-1].output

        self.L1 = sum([layer.L1 for layer in self.layers])

        self.outdim = (batch_size, nkerns[-1], h, w)

        self.params = [param for layer in self.layers for param in layer.params]


def test_mlp(init_learning_rate=.1,
             init_momentum=.05,
             n_epochs=10000,
             dataset='./data/ubiquitous.pkl',
             batch_size=20,
             nodenums=[4*400, 400, 20, 2],
             names=['fc0', 'fc1', 'softmax'],
             ps=[0.5, 0.5]):
    # load dataset
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

    classifier = DMLP(rng=rng,
                      names=names,
                      is_train=is_train,
                      input=x,
                      nodenums=nodenums,
                      ps=ps)

    cost = classifier.negative_log_likelihood(y)

    test_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size],
                is_train: np.cast['int32'](0)})

    validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size],
                is_train: np.cast['int32'](0)})

    learning_rate = theano.shared(np.cast[theano.config.floatX](init_learning_rate))

    assert init_momentum >= 0. and init_momentum < 1.

    momentum = theano.shared(np.cast[theano.config.floatX](init_momentum), name='momentum')
    updates = []
    for param in classifier.params:
        param_update = theano.shared(param.get_value()*np.cast[theano.config.floatX](0.))
        updates.append((param,
            param - learning_rate * param_update))
        updates.append((param_update,
            momentum*param_update +
            (np.cast[theano.config.floatX](1.)-momentum)*T.grad(cost, param)))

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
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience/2)

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        print "momentum: {}".format(momentum.get_value())
        print "learning_rate: {}".format(learning_rate.get_value())
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:

                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

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


def test_cnn(init_learning_rate=.1,
             init_momentum=.05,
             n_epochs=10000,
             dataset='./data/ubiquitous.pkl',
             h=4,
             w=400,
             batch_size=500,
             convnames=['conv0', 'conv1'],
             nkerns=[400, 200],
             filtersizes=[(4,12), (1,10)],
             poolsizes=[(1,100), (1,10)],
             strides=[(1,20), (1,2)],
             nodenums=[400, 20, 2],
             dmlpnames=['fc0', 'fc1', 'softmax'],
             ps=[0.5, 0.5],
             l1_reg=0.001):
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

    cost = classifier.negative_log_likelihood(y) + l1_reg * extractfeat.L1

    test_model = theano.function(
            inputs=[index],
            outputs=[classifier.errors(y), classifier.p_y_given_x, classifier.y_pred],
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size],
                is_train: np.cast['int32'](0)})

    validate_model = theano.function(
            inputs=[index],
            outputs=[classifier.errors(y), classifier.p_y_given_x, classifier.y_pred],
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size],
                is_train: np.cast['int32'](0)})


    learning_rate = theano.shared(np.cast[theano.config.floatX](init_learning_rate))

    assert init_momentum >= 0. and init_momentum < 1.

    momentum = theano.shared(np.cast[theano.config.floatX](init_momentum), name='momentum')
    updates = []
    for param in extractfeat.params +  classifier.params:
        param_update = theano.shared(param.get_value()*np.cast[theano.config.floatX](0.))
        updates.append((param,
            param - learning_rate * param_update))
        updates.append((param_update,
            momentum*param_update +
            (np.cast[theano.config.floatX](1.)-momentum)*T.grad(cost, param)))

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
    patience = 10000
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

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    # test model
                    test_losses = [
                        test_model(i)[0]
                        for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print(('[Test] epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    test_py = [
                            test_model(i)[1]
                            for i in xrange(n_test_batches)]
                    y_scores = np.array([a[1] for batch in test_py for a in batch])
                    # print y_true, y_true.shape
                    # print y_scores, y_scores.shape
                    fpr, tpr, _ = metrics.roc_curve(y_true, y_scores)
                    precision, recall, _ = metrics.precision_recall_curve(y_true, y_scores)
                    print '    ROC AUC score is {}'.format(metrics.roc_auc_score(y_true, y_scores))
                    # print '    Precision score is {}'.format(metrics.precision_score(y_true, y_pred))

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


if __name__ == '__main__':
    # evaluate_cnn()
    # test_mlp(dataset='../DeepLearningTutorials/data/mnist.pkl.gz',
    #          names=['fc0', 'softmax'],
    #          nodenums=[28*28, 500, 10],
    #          ps=[0.5])
    # test_cnn(init_learning_rate=.1,
    #          init_momentum=.05,
    #          n_epochs=10000,
    #          dataset='../DeepLearningTutorials/data/mnist.pkl.gz',
    #          h=28,
    #          w=28,
    #          batch_size=100,
    #          convnames=['conv0', 'conv1'],
    #          nkerns=[100, 20],
    #          filtersizes=[(4,4), (4,4)],
    #          poolsizes=[(2,2), (2,2)],
    #          strides=[(2,2), (2,2)],
    #          nodenums=[100, 20, 10],
    #          dmlpnames=['fc0', 'fc1', 'softmax'],
    #          ps=[0.9, 0.9])
    test_cnn(batch_size=400,
             n_epochs=20000,
             convnames=['conv0'],
             nkerns=[600],
             filtersizes=[(4,15)],
             poolsizes=[(1,50)],
             strides=[(1,10)],
             dmlpnames=['fc0', 'softmax'],
             nodenums=[200,2],
             ps=[0.5],
             l1_reg=0.05
             )
