"""Educational float64 NumPy layers with explicit backward passes."""
import numpy as np


def array(x):
    if np.iscomplexobj(x):
        raise ValueError('Real arrays only')
    a = np.asarray(x, dtype=float)
    if not a.size or not np.isfinite(a).all():
        raise ValueError('Nonempty finite input required')
    return a


def positive_int(n):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('Positive integer required')
    return int(n)


def exact_gradient(g, shape):
    g = array(g)
    if g.shape != shape:
        raise ValueError('Gradient shape mismatch')
    return g


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
    if not np.isfinite(loss):
        raise FloatingPointError('Nonfinite cross entropy')
    g = softmax(x)
    g[np.arange(len(y)), y] -= 1
    return float(loss), g / len(y)


def mse(pred, target):
    p, t = array(pred), array(target)
    if p.shape != t.shape:
        raise ValueError('Target shape mismatch')
    with np.errstate(over='raise', invalid='raise'):
        d = p - t
        loss, g = float(np.mean(d*d)), 2*d/d.size
    return loss, g


class Dense:
    def __init__(self, inputs, outputs, rng, scale=None):
        inputs, outputs = positive_int(inputs), positive_int(outputs)
        if scale is not None and (not np.isfinite(scale) or scale < 0):
            raise ValueError('Invalid initialization scale')
        self.w = rng.normal(0, np.sqrt(2/inputs) if scale is None else scale, (inputs, outputs))
        self.b = np.zeros(outputs)

    def forward(self, x, training=False):
        x = array(x)
        if x.ndim != 2 or x.shape[1] != self.w.shape[0]:
            raise ValueError('Expected batch by input features')
        self.x = x.copy()
        return self.x @ self.w + self.b

    def backward(self, g):
        g = exact_gradient(g, (len(self.x), self.w.shape[1]))
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
        self.x = array(x).copy()
        self.y = {'relu': lambda: np.maximum(self.x, 0), 'tanh': lambda: np.tanh(self.x), 'sigmoid': lambda: sigmoid(self.x)}[self.kind]()
        return self.y

    def backward(self, g):
        g = exact_gradient(g, self.x.shape)
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
        x = array(x)
        self.mask = (self.rng.random(x.shape) >= self.p)/(1-self.p) if training else np.ones_like(x)
        return x*self.mask

    def backward(self, g):
        return exact_gradient(g, self.mask.shape)*self.mask

    def parameters(self):
        return []


class Conv2D:
    """Valid cross-correlation, NCHW, stride 1, no padding."""
    def __init__(self, channels, filters, kernel, rng):
        channels, filters, kernel = map(positive_int, (channels, filters, kernel))
        self.w = rng.normal(0, np.sqrt(2/(channels*kernel**2)), (filters, channels, kernel, kernel))
        self.b = np.zeros(filters)

    def forward(self, x, training=False):
        self.x = array(x).copy()
        k = self.w.shape[-1]
        if self.x.ndim != 4 or self.x.shape[1] != self.w.shape[1] or min(self.x.shape[2:]) < k:
            raise ValueError('Invalid NCHW input')
        self.windows = np.lib.stride_tricks.sliding_window_view(self.x, (k,k), axis=(2,3))
        return np.einsum('nchwij,fcij->nfhw', self.windows, self.w) + self.b[None,:,None,None]

    def backward(self, g):
        g = exact_gradient(g, (len(self.x), len(self.w), *self.windows.shape[2:4]))
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
        x = array(x)
        if x.ndim < 2:
            raise ValueError('Expected a batch dimension and features')
        self.shape=x.shape
        return x.reshape(len(x),-1)
    def backward(self,g):
        return exact_gradient(g, (self.shape[0], int(np.prod(self.shape[1:])))).reshape(self.shape)
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
        # Validate and stage every update before mutating any parameter or state.
        t = self.t + 1
        pending = []
        seen = set()
        for p,g in params:
            if (not isinstance(p, np.ndarray) or p.dtype != np.float64 or not p.flags.writeable
                    or not p.size or not np.isfinite(p).all() or id(p) in seen
                    or any(np.shares_memory(p, old) for old, *_ in pending)):
                raise ValueError('Expected distinct writable finite float64 parameters')
            seen.add(id(p))
            g = exact_gradient(g, p.shape)
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                g = g + self.l2*p
                m,v=self.state.get(id(p),(np.zeros_like(p),np.zeros_like(p)))
                if self.method=='momentum':
                    m=.9*m+g; update=m
                elif self.method=='adam':
                    m=.9*m+.1*g; v=.999*v+.001*g*g
                    update=(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
                else: update=g
                candidate = p - self.rate*update
            if not all(np.isfinite(a).all() for a in (candidate, update, m, v)):
                raise FloatingPointError('Nonfinite update')
            pending.append((p,candidate,m,v))
        if not pending:
            raise ValueError('No parameters supplied')
        for p,candidate,m,v in pending:
            p[...] = candidate
            self.state[id(p)]=(m,v)
        self.t=t


def perceptron(x,y,epochs=20):
    epochs=positive_int(epochs)
    x=array(x); y=np.asarray(y)
    if x.ndim!=2 or y.shape!=(len(x),) or not np.isin(y,[0,1]).all(): raise ValueError('Invalid binary data')
    w=np.zeros(x.shape[1]); b=0.
    for _ in range(epochs):
        for a,t in zip(x,y):
            error=t-int(a@w+b>=0); w+=error*a; b+=error
    return w,b


def train(model,x,y,optimizer,epochs=100,batch=32,seed=42):
    epochs, batch = positive_int(epochs), positive_int(batch)
    x, y = array(x), np.asarray(y)
    if x.ndim < 2 or y.shape != (len(x),) or y.dtype.kind not in 'iu' or (y < 0).any():
        raise ValueError('Invalid training samples or labels')
    # Validate all labels before any optimizer update, not just the current batch.
    cross_entropy(model.forward(x), y)
    rng=np.random.default_rng(seed); history=[]
    for _ in range(epochs):
        order=rng.permutation(len(x))
        for i in range(0,len(x),batch):
            idx=order[i:i+batch]; _,g=cross_entropy(model.forward(x[idx],True),y[idx]); model.backward(g); optimizer.step(model.parameters())
        history.append(cross_entropy(model.forward(x),y)[0])
    return np.asarray(history)
