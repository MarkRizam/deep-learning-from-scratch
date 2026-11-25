import numpy as np
import pytest
import torch
from dl_from_scratch import *

@pytest.mark.parametrize('seed',[11,22,33])
def test_dense(seed):
    r=np.random.default_rng(seed); x=r.normal(size=(5,4)); y=r.integers(3,size=5); l=Dense(4,3,r); z=l.forward(x); loss,g=cross_entropy(z,y); dx=l.backward(g)
    a=torch.tensor(x,requires_grad=True); w=torch.tensor(l.w,requires_grad=True); b=torch.tensor(l.b,requires_grad=True); ref=torch.nn.functional.cross_entropy(a@w+b,torch.tensor(y)); ref.backward()
    assert loss==pytest.approx(ref.item()); np.testing.assert_allclose(dx,a.grad.numpy(),atol=1e-12); np.testing.assert_allclose(l.dw,w.grad.numpy(),atol=1e-12); np.testing.assert_allclose(l.db,b.grad.numpy(),atol=1e-12)

@pytest.mark.parametrize('kind',['relu','sigmoid','tanh'])
def test_activation(kind):
    x=np.array([-2.,-.4,.3,2.]); layer=Activation(kind); layer.forward(x); g=layer.backward(np.ones_like(x)); h=1e-5
    np.testing.assert_allclose(g,(layer.forward(x+h)-layer.forward(x-h))/(2*h),atol=1e-9)

@pytest.mark.parametrize('seed',[11,22,33])
def test_conv(seed):
    r=np.random.default_rng(seed); x=r.normal(size=(2,2,5,6)); l=Conv2D(2,3,3,r); z=l.forward(x); g=r.normal(size=z.shape); dx=l.backward(g)
    a=torch.tensor(x,requires_grad=True); w=torch.tensor(l.w,requires_grad=True); b=torch.tensor(l.b,requires_grad=True); ref=torch.nn.functional.conv2d(a,w,b); ref.backward(torch.tensor(g))
    for p,q in [(z,ref.detach()),(dx,a.grad),(l.dw,w.grad),(l.db,b.grad)]: np.testing.assert_allclose(p,q.numpy(),atol=1e-11)

@pytest.mark.parametrize('method',['sgd','momentum','adam'])
def test_optimizer(method):
    p=np.array([2.,-1.]); opt=Optimizer(method,.03)
    for _ in range(500): opt.step([(p,2*p)])
    assert p@p<1e-7

@pytest.mark.parametrize('seed',[11,22,33])
def test_xor(seed):
    r=np.random.default_rng(seed); x=np.array([[0.,0.],[0,1],[1,0],[1,1]]); y=np.array([0,1,1,0]); m=Sequential(Dense(2,8,r),Activation('tanh'),Dense(8,2,r)); h=train(m,x,y,Optimizer('adam',.05),400,4,seed)
    assert h[-1]<.05; assert np.array_equal(m.forward(x).argmax(1),y)

def test_perceptron():
    x=np.array([[0.,0.],[0,1],[1,0],[1,1]]); y=np.array([0,0,0,1]); w,b=perceptron(x,y); assert np.array_equal(x@w+b>=0,y)

def test_stable_softmax():
    x=np.array([[1000.,1001.],[-1000.,-1001.]]); assert np.isfinite(cross_entropy(x,[0,1])[0]); np.testing.assert_allclose(softmax(x).sum(1),1)

def test_dropout():
    l=Dropout(.3,np.random.default_rng(11)); x=np.ones(10000); np.testing.assert_array_equal(l.forward(x),x); y=l.forward(x,True); assert .95<y.mean()<1.05; np.testing.assert_array_equal(l.backward(x),y)

def test_mse():
    loss,g=mse([1,2],[3,4]); assert loss==4; np.testing.assert_array_equal(g,[-2,-2])

def test_l2():
    p=np.array([1.]); Optimizer('sgd',.1,.5).step([(p,np.zeros(1))]); assert p[0]==pytest.approx(.95)

@pytest.mark.parametrize('f',[lambda:softmax([1,2]),lambda:cross_entropy([[1,2]],[2]),lambda:cross_entropy([[1,2]],[.5]),lambda:mse([1],[1,2]),lambda:Activation('bad'),lambda:Dropout(1,np.random.default_rng(1)),lambda:Optimizer(rate=0),lambda:Conv2D(2,3,3,np.random.default_rng(1)).forward(np.zeros((1,1,5,5))),lambda:sigmoid([np.nan]),lambda:sigmoid(np.array([1j]))])
def test_invalid(f):
    with pytest.raises(ValueError): f()
