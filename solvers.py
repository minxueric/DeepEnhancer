#!/usr/bin/env python
# encoding: utf-8

from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T


def sgd(loss, params, learning_rate):
    grads = T.grad(loss, params)
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad
    return updates


def apply_momentum(updates, momentum):
    params = updates.keys()
    updates = OrderedDict(updates)
    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(
                np.zeros(value.shape, dtype=value.dtype),
                broadcastable=param.broadcastable)
        vplusparam = momentum * velocity + updates[param]
        updates[velocity] = vplusparam - param
        updates[param] = vplusparam
    return updates


def sgd_momentum(loss, params, learning_rate, momentum):
    updates = sgd(loss, params, learning_rate)
    return apply_momentum(updates, momentum)


def apply_nesterov_momentum(updates, momentum):
    params = updates.keys()
    updates = OrderedDict(updates)
    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(
                np.zeros(value.shape, dtype=value.dtype),
                broadcastable=param.broadcastable)
        v = momentum * velocity + (updates[param] - param)
        updates[velocity] = v
        updates[param] = momentum * v + updates[param]
    return updates


def nesterov_momentum(loss, params, learning_rate, momentum):
    updates = sgd(loss, params, learning_rate)
    return apply_nesterov_momentum(updates, momentum)


def adagrad(loss, params, learning_rate, epsilon=1e-6):
    grads = T.grad(loss, params)
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(
                np.zeros(value.shape, dtype=value.dtype),
                broadcastable=param.broadcastable)
        accu_new = accu + grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                T.sqrt(accu_new + epsilon))
    return updates


def rmsprop(loss, params, learning_rate, rho=0.9, epsilon=1e-6):
    grads = T.grad(loss, params)
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(
                np.zeros(value.shape, dtype=value.dtype),
                broadcastable=param.broadcastable)
        accu_new = rho * accu + (1 - rho) * grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                T.sqrt(accu_new + epsilon))
    return updates


def adadelta(loss, params, learning_rate, rho=.95, epsilon=1e-6):
    grads = T.grad(loss, params)
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(
                np.zeros(value.shape, dtype=value.dtype),
                broadcastable=param.broadcastable)
        delta_accu = theano.shared(
                np.zeros(value.shape, dtype=value.dtype),
                broadcastable=param.broadcastable)
        accu_new = rho * accu + (1 - rho) * grad ** 2
        updates[accu] = accu_new
        update = (grad * T.sqrt(delta_accu + epsilon) /
                T.sqrt(accu_new + epsilon))
        updates[param] = param - learning_rate * update
        delta_accu_new = rho * delta_accu + (1 - rho) * update ** 2
        updates[delta_accu] = delta_accu_new
    return updates


def adam(loss, params, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    grads = T.grad(loss, params)
    updates = OrderedDict()
    t_prev = theano.shared(np.cast[theano.config.floatX](0))
    t = t_prev + 1
    a_t = learning_rate * T.sqrt(1-beta2**t)/(1-beta1**t)
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(
                np.zeros(value.shape, dtype=value.dtype),
                broadcastable=param.broadcastable)
        v_prev = theano.shared(
                np.zeros(value.shape, dtype=value.dtype),
                broadcastable=param.broadcastable)
        m_t = beta1 * m_prev + (1 - beta1) * grad
        v_t = beta2 * v_prev + (1 - beta2) * grad ** 2
        step = a_t * m_t / (T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step
    updates[t_prev] = t
    return updates
