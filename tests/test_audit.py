"""Regression cases from AIR-004 V1.1 audit."""
import numpy as np
import pytest
from dl_from_scratch import Activation, Conv2D, Dense, Dropout, Flatten, Optimizer, Sequential, cross_entropy, mse, perceptron, train


@pytest.mark.parametrize('factory,x,g', [
    (lambda r: Dense(2,3,r), np.ones((2,2)), np.ones(3)),
    (lambda r: Activation('relu'), np.ones((2,3)), np.ones(3)),
    (lambda r: Dropout(.2,r), np.ones((2,3)), np.ones(3)),
    (lambda r: Conv2D(1,2,2,r), np.ones((1,1,3,3)), np.ones((1,2,1,2))),
    (lambda r: Flatten(), np.ones((2,2,3)), np.ones((3,4))),
])
def test_exact_gradient_shape(factory,x,g):
    layer=factory(np.random.default_rng(1)); layer.forward(x)
    with pytest.raises(ValueError): layer.backward(g)


@pytest.mark.parametrize('fn', [
    lambda r: Dense(0,2,r), lambda r: Dense(2,True,r),
    lambda r: Dense(2,3,r,np.nan), lambda r: Conv2D(1,2,0,r),
    lambda r: Dense(2,3,r).forward(np.ones((2,2,2))),
    lambda r: Flatten().forward(np.ones(3)),
    lambda r: perceptron([[0],[1]],[0,1],0),
    lambda r: train(Sequential(Dense(1,2,r)),[[0],[1]],[0,1],Optimizer(),epochs=0),
    lambda r: train(Sequential(Dense(1,2,r)),[[0],[1]],[0,1],Optimizer(),batch=True),
])
def test_invalid_configuration(fn):
    with pytest.raises(ValueError): fn(np.random.default_rng(1))


@pytest.mark.parametrize('method',['sgd','momentum','adam'])
def test_optimizer_failed_step_is_atomic(method):
    p=np.ones((2,3)); q=np.ones(2); opt=Optimizer(method)
    with pytest.raises(ValueError): opt.step([(p,np.ones_like(p)),(q,np.ones(1))])
    np.testing.assert_array_equal(p,np.ones_like(p)); np.testing.assert_array_equal(q,np.ones_like(q))
    assert opt.t==0 and not opt.state
    opt.step([(p,np.ones_like(p))]); assert opt.t==1


def test_optimizer_rejects_alias():
    p=np.ones(3); opt=Optimizer()
    with pytest.raises(ValueError): opt.step([(p,np.ones_like(p)),(p[:2],np.ones(2))])
    np.testing.assert_array_equal(p,np.ones(3)); assert opt.t==0


def test_training_invalid_label_before_updates():
    model=Sequential(Dense(1,2,np.random.default_rng(1))); opt=Optimizer()
    with pytest.raises(ValueError): train(model,[[0],[1]],[0,3],opt,epochs=1,batch=1)
    assert opt.t==0


def test_dense_input_cache():
    layer=Dense(2,1,np.random.default_rng(1)); x=np.ones((1,2)); layer.forward(x); x[:]=3
    layer.backward(np.ones((1,1))); np.testing.assert_array_equal(layer.dw,np.ones((2,1)))


@pytest.mark.parametrize('fn',[lambda:mse([1e200],[0]),lambda:cross_entropy([[1e308,-1e308]],[1])])
def test_loss_overflow(fn):
    with pytest.raises(FloatingPointError): fn()


def test_adam_overflow_is_atomic():
    p=np.ones(1); opt=Optimizer('adam')
    with pytest.raises(FloatingPointError): opt.step([(p,np.array([1e200]))])
    np.testing.assert_array_equal(p,np.ones(1)); assert opt.t==0 and not opt.state
