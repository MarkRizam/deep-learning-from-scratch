"""Educational float64 NumPy layers with explicit backward passes."""
import numpy as np


def array(x):
    if np.iscomplexobj(x):
        raise ValueError('Real arrays only')
    a = np.asarray(x, dtype=float)
    if not a.size or not np.isfinite(a).all():
        raise ValueError('Nonempty finite input required')
    return a


def sigmoid(x):
    x = array(x)
    return np.exp(-np.logaddexp(0, -x))


def softmax(x):
    x = array(x)
    if x.ndim != 2:
        raise ValueError('Expected batch by classes')
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits, labels):
    x = array(logits)
    y = np.asarray(labels)
    if x.ndim != 2 or y.shape != (len(x),) or y.dtype.kind not in 'iu' or (y < 0).any() or (y >= x.shape[1]).any():
        raise ValueError('Invalid logits or integer class labels')
    z = x - x.max(axis=1, keepdims=True)
    loss = np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y])
    g = softmax(x)
    g[np.arange(len(y)), y] -= 1
    return float(loss), g / len(y)


def mse(pred, target):
    p, t = array(pred), array(target)
    if p.shape != t.shape:
        raise ValueError('Target shape mismatch')
    d = p - t
    return float(np.mean(d*d)), 2*d/d.size


class Dense:
    def __init__(self, inputs, outputs, rng, scale=None):
        self.w = rng.normal(0, np.sqrt(2/inputs) if scale is None else scale, (inputs, outputs))
        self.b = np.zeros(outputs)

    def forward(self, x, training=False):
        self.x = array(x)
        return self.x @ self.w + self.b

    def backward(self, g):
        self.dw = self.x.T @ g
        self.db = g.sum(0)
        return g @ self.w.T

    def parameters(self):
        return [(self.w, self.dw), (self.b, self.db)]


class Activation:
    def __init__(self, kind):
        if kind not in ('relu', 'tanh', 'sigmoid'):
            raise ValueError('Unknown activation')
        self.kind = kind

    def forward(self, x, training=False):
        self.x = array(x)
        self.y = {'relu': lambda: np.maximum(self.x, 0), 'tanh': lambda: np.tanh(self.x), 'sigmoid': lambda: sigmoid(self.x)}[self.kind]()
        return self.y

    def backward(self, g):
        d = {'relu': lambda: self.x > 0, 'tanh': lambda: 1-self.y**2, 'sigmoid': lambda: self.y*(1-self.y)}[self.kind]()
        return g*d

    def parameters(self):
        return []


class Dropout:
    def __init__(self, p, rng):
        if not 0 <= p < 1:
            raise ValueError('Expected 0 <= p < 1')
        self.p, self.rng = p, rng

    def forward(self, x, training=False):
        self.mask = (self.rng.random(x.shape) >= self.p)/(1-self.p) if training else np.ones_like(x)
        return x*self.mask

    def backward(self, g):
        return g*self.mask

    def parameters(self):
        return []


class Conv2D:
    """Valid cross-correlation, NCHW, stride 1, no padding."""
    def __init__(self, channels, filters, kernel, rng):
        self.w = rng.normal(0, np.sqrt(2/(channels*kernel**2)), (filters, channels, kernel, kernel))
        self.b = np.zeros(filters)

    def forward(self, x, training=False):
        self.x = array(x)
        k = self.w.shape[-1]
        if self.x.ndim != 4 or self.x.shape[1] != self.w.shape[1] or min(self.x.shape[2:]) < k:
            raise ValueError('Invalid NCHW input')
        self.windows = np.lib.stride_tricks.sliding_window_view(self.x, (k,k), axis=(2,3))
        return np.einsum('nchwij,fcij->nfhw', self.windows, self.w) + self.b[None,:,None,None]

    def backward(self, g):
        self.dw = np.einsum('nchwij,nfhw->fcij', self.windows, g)
        self.db = g.sum((0,2,3))
        dx = np.zeros_like(self.x)
        h,w = g.shape[2:]
        for i in range(self.w.shape[-1]):
            for j in range(self.w.shape[-1]):
                dx[:,:,i:i+h,j:j+w] += np.einsum('nfhw,fc->nchw',g,self.w[:,:,i,j])
        return dx

    def parameters(self):
        return [(self.w,self.dw),(self.b,self.db)]


class Flatten:
    def forward(self,x,training=False):
        self.shape=x.shape
        return x.reshape(len(x),-1)
    def backward(self,g):
        return g.reshape(self.shape)
    def parameters(self):
        return []


class Sequential:
    def __init__(self,*layers): self.layers=layers
    def forward(self,x,training=False):
        for layer in self.layers: x=layer.forward(x,training)
        return x
    def backward(self,g):
        for layer in reversed(self.layers): g=layer.backward(g)
        return g
    def parameters(self):
        return [pair for layer in self.layers for pair in layer.parameters()]


class Optimizer:
    def __init__(self, method='sgd', rate=.01, l2=0.):
        if method not in ('sgd','momentum','adam') or not np.isfinite(rate) or rate<=0 or not np.isfinite(l2) or l2<0:
            raise ValueError('Invalid optimizer')
        self.method,self.rate,self.l2=method,rate,l2
        self.state={}; self.t=0
    def step(self, params):
        self.t+=1
        for p,g in params:
            g=array(g)+self.l2*p
            if g.shape!=p.shape: raise ValueError('Gradient shape mismatch')
            m,v=self.state.get(id(p),(np.zeros_like(p),np.zeros_like(p)))
            if self.method=='momentum':
                m=.9*m+g; update=m
            elif self.method=='adam':
                m=.9*m+.1*g; v=.999*v+.001*g*g
                update=(m/(1-.9**self.t))/(np.sqrt(v/(1-.999**self.t))+1e-8)
            else: update=g
            if not np.isfinite(update).all() or not np.isfinite(m).all() or not np.isfinite(v).all(): raise FloatingPointError('Nonfinite update')
            p-=self.rate*update
            if not np.isfinite(p).all(): raise FloatingPointError('Nonfinite parameter')
            self.state[id(p)]=(m,v)


def perceptron(x,y,epochs=20):
    x=array(x); y=np.asarray(y)
    if x.ndim!=2 or y.shape!=(len(x),) or not np.isin(y,[0,1]).all(): raise ValueError('Invalid binary data')
    w=np.zeros(x.shape[1]); b=0.
    for _ in range(epochs):
        for a,t in zip(x,y):
            error=t-int(a@w+b>=0); w+=error*a; b+=error
    return w,b


def train(model,x,y,optimizer,epochs=100,batch=32,seed=42):
    rng=np.random.default_rng(seed); history=[]
    for _ in range(epochs):
        order=rng.permutation(len(x))
        for i in range(0,len(x),batch):
            idx=order[i:i+batch]; _,g=cross_entropy(model.forward(x[idx],True),y[idx]); model.backward(g); optimizer.step(model.parameters())
        history.append(cross_entropy(model.forward(x),y)[0])
    return np.asarray(history)
