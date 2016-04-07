import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T
from layers import ReLU, load_data, DMLP
from solvers import sgd_momentum, nesterov_momentum

class myDMLP(DMLP):
    def __init__(self, names, nodenums, ps, activation=ReLU):
        print '... initializing the model'
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.is_train = T.iscalar('is_train')
        rng = np.random.RandomState(23456)
        DMLP.__init__(self, rng, names, self.is_train, self.x, self.y, nodenums, ps, activation)
        # cost function to be optimized
        self.cost = self.negative_log_likelihood

    def load(self, file, batch_size):
        print '... loading the data %s' % file
        datasets = load_data(file)
        self.batch_size = batch_size
        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]
        self.test_set_x, self.test_set_y = datasets[2]
        self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0]
        self.n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0]
        self.n_test_batches = self.test_set_x.get_value(borrow=True).shape[0]
        self.n_train_batches /= batch_size
        self.n_valid_batches /= batch_size
        self.n_test_batches /= batch_size

    def train(self, solver, init_learning_rate, init_momentum, n_epochs):
        print '... building the model'
        index = T.lscalar()
        batch_size = self.batch_size
        self.test_model = theano.function(
                inputs=[index],
                outputs=self.errors,
                givens={
                    self.x: self.test_set_x[index * batch_size : (index + 1) * batch_size],
                    self.y: self.test_set_y[index * batch_size : (index + 1) * batch_size],
                    self.is_train: np.cast['int32'](0)})
        self.validate_model = theano.function(
                inputs=[index],
                outputs=self.errors,
                givens={
                    self.x: self.valid_set_x[index * batch_size : (index + 1) * batch_size],
                    self.y: self.valid_set_y[index * batch_size : (index + 1) * batch_size],
                    self.is_train: np.cast['int32'](0)})
        self.learning_rate = theano.shared(
                value=np.cast[theano.config.floatX](init_learning_rate),
                name = 'learning_rate')
        self.momentum = theano.shared(
                value=np.cast[theano.config.floatX](init_momentum),
                name='momentum')
        if solver in [sgd_momentum, nesterov_momentum]:
            updates = solver(self.cost, self.params, self.learning_rate, self.momentum)
        else:
            updates = solver(self.cost, self.params, self.learning_rate)
        self.train_model = theano.function(
                inputs=[index],
                outputs=self.cost,
                updates=updates,
                givens={
                    self.x: self.train_set_x[index * batch_size : (index + 1) * batch_size],
                    self.y: self.train_set_y[index * batch_size : (index + 1) * batch_size],
                    self.is_train: np.cast['int32'](1)})

        print '... training the model'
        patience = 1000
        patience_increase = 2
        improvement_threshold = 0.995
        validation_frequency = min(self.n_train_batches, patience/2)

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            print 'momentum:\t{}'.format(self.momentum.get_value())
            print 'learning_rate:\t{}'.format(self.learning_rate.get_value())
            for batchindex in xrange(self.n_train_batches):
                iternum = (epoch - 1) * self.n_train_batches + batchindex
                if iternum % 100 == 0:
                    print 'training @ iter = {}'.format(iternum)
                cost_ij = self.train_model(batchindex)
                if (iternum + 1) % validation_frequency == 0:
                    validation_losses = [self.validate_model(i) for i in xrange(self.n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    print '[Validation] epoch {}, minibatch {}/{}, validation error {}%%'.format(
                            epoch, batchindex+1, self.n_train_batches, this_validation_loss * 100.)
                    if this_validation_loss < best_validation_loss:
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iternum * patience_increase)
                        best_validation_loss = this_validation_loss
                        best_iter = iternum
                        test_losses = [self.test_model(i) for i in xrange(self.n_test_batches)]
                        test_score = np.mean(test_losses)
                        print '[Test] epoch {}, minibatch {}/{}, test error {}%%'.format(
                                epoch, batchindex + 1, self.n_train_batches, test_score * 100.)
                if patience <= iternum:
                    done_looping = True
                    break
            if self.momentum.get_value() < .99:
                new_momentum = 1. - ( 1. - self.momentum.get_value()) * .98
                self.momentum.set_value(np.cast[theano.config.floatX](new_momentum))
            new_learning_rate = self.learning_rate.get_value() * .99
            self.learning_rate.set_value(np.cast[theano.config.floatX](new_learning_rate))
        end_time = timeit.default_timer()
        print 'Optimization complete.'
        print 'Best validation score of {} %% obtained at iter {},\n with test performance {} %%.'.format(
                best_validation_loss * 100., best_iter + 1, test_score * 100.)
        print >> sys.stderr, ('The code for file' + os.path.split(__file__)[1]+
                ' ran for %.2fm' %((end_time - start_time) / 60.))


if __name__ == '__main__':
    classifier = myDMLP(
            names=['fc0', 'softmax'],
            nodenums=[28*28, 500, 10],
            ps=[0.5])
    classifier.load(
            file='../DeepLearningTutorials/data/mnist.pkl.gz',
            batch_size=400)
    classifier.train(
            solver=nesterov_momentum,
            init_learning_rate=0.1,
            init_momentum=0.05,
            n_epochs=50)
