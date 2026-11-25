"""Eight fixed CPU synthetic experiments; PyTorch is a reference only."""
import json
import platform
from pathlib import Path
import numpy as np
from .core import *

SEEDS=(11,22,33)
TITLES=['Dense gradient agreement','Activation saturation','XOR learning','Optimizer convergence','Initialization sensitivity','Dropout and L2','Convolution gradient agreement','Synthetic image classification']


def run():
    import torch
    torch.set_num_threads(1)
    rows=[]
    def add(e,seed,c,m,a,b,ref):
        rows.append(dict(experiment=f'EXP-{e:03}',seed=seed,configuration=c,metric=m,ours=float(a),reference=float(b),reference_kind=ref))
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        x=rng.normal(size=(7,3)); y=rng.integers(2,size=7); dense=Dense(3,2,rng)
        logits=dense.forward(x); loss,g=cross_entropy(logits,y); dx=dense.backward(g)
        tx=torch.tensor(x,requires_grad=True); tw=torch.tensor(dense.w,requires_grad=True); tb=torch.tensor(dense.b,requires_grad=True)
        tl=torch.nn.functional.cross_entropy(tx@tw+tb,torch.tensor(y)); tl.backward()
        err=max(abs(loss-tl.item()),np.max(abs(dense.dw-tw.grad.numpy())),np.max(abs(dense.db-tb.grad.numpy())),np.max(abs(dx-tx.grad.numpy())))
        add(1,seed,'dense + cross entropy','maximum absolute difference',err,0,'PyTorch agreement ideal')
        eps=1e-5; w=dense.w[0,0]; dense.w[0,0]=w+eps; a=cross_entropy(dense.forward(x),y)[0]; dense.w[0,0]=w-eps; b=cross_entropy(dense.forward(x),y)[0]; dense.w[0,0]=w
        add(1,seed,'central difference','absolute derivative error',abs((a-b)/(2*eps)-dense.dw[0,0]),0,'analytic backprop agreement ideal')
        x=rng.normal(size=(1000,))*3
        for act in ('sigmoid','tanh','relu'):
            layer=Activation(act); layer.forward(x); d=layer.backward(np.ones_like(x))
            add(2,seed,act,'fraction derivative below 0.01',np.mean(abs(d)<.01),0,'zero saturation ideal (not competing method)')
        x=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]); y=np.array([0,1,1,0]); w,b=perceptron(x,y)
        p=np.mean((x@w+b>=0)==y)
        model=Sequential(Dense(2,8,rng),Activation('tanh'),Dense(8,2,rng)); train(model,x,y,Optimizer('adam',.05),epochs=400,batch=4,seed=seed)
        add(3,seed,'XOR','training accuracy',np.mean(model.forward(x).argmax(1)==y),p,'single perceptron')
        x=rng.normal(size=(240,4)); y=(x[:,0]*x[:,1]+.4*x[:,2]>0).astype(int); tr,te=x[:180],x[180:]; yt,ye=y[:180],y[180:]
        def mlp(scale=None,drop=0):
            r=np.random.default_rng(seed); return Sequential(Dense(4,16,r,scale),Activation('tanh'),Dropout(drop,r),Dense(16,2,r,scale))
        for method in ('sgd','momentum','adam'):
            m=mlp(); h=train(m,tr,yt,Optimizer(method,.02),epochs=50,batch=30,seed=seed)
            add(4,seed,method,'training cross entropy',h[-1],h[0],'first epoch loss')
        for scale in (.01,.5,2.):
            m=mlp(scale); h=train(m,tr,yt,Optimizer('adam',.02),epochs=40,batch=30,seed=seed)
            add(5,seed,f'scale={scale}','training cross entropy',h[-1],h[0],'first epoch loss')
        base=mlp(); train(base,tr,yt,Optimizer('adam',.02),epochs=60,batch=30,seed=seed); baseline=np.mean(base.forward(te).argmax(1)==ye)
        for drop,l2 in ((0.,0.),(.3,0.),(0.,.01),(.3,.01)):
            m=mlp(drop=drop); train(m,tr,yt,Optimizer('adam',.02,l2),epochs=60,batch=30,seed=seed)
            add(6,seed,f'dropout={drop}, L2={l2}','held-out accuracy',np.mean(m.forward(te).argmax(1)==ye),baseline,'unregularized matched initialization')
        x=rng.normal(size=(2,2,5,5)); conv=Conv2D(2,3,3,rng); out=conv.forward(x); g=rng.normal(size=out.shape); dx=conv.backward(g)
        tx=torch.tensor(x,requires_grad=True); tw=torch.tensor(conv.w,requires_grad=True); tb=torch.tensor(conv.b,requires_grad=True); out0=torch.nn.functional.conv2d(tx,tw,tb); out0.backward(torch.tensor(g))
        for name,a,b in [('forward',out,out0.detach().numpy()),('input gradient',dx,tx.grad.numpy()),('weight gradient',conv.dw,tw.grad.numpy()),('bias gradient',conv.db,tb.grad.numpy())]:
            add(7,seed,name,'maximum absolute difference',np.max(abs(a-b)),0,'PyTorch agreement ideal')
        y=rng.integers(2,size=160); x=rng.normal(0,.65,(160,1,6,6));
        for i,label in enumerate(y):
            pos=rng.integers(1,5)
            if label==0: x[i,0,pos,:]+=1.5
            else: x[i,0,:,pos]+=1.5
        r=np.random.default_rng(seed); cnn=Sequential(Conv2D(1,4,3,r),Activation('relu'),Flatten(),Dense(64,2,r)); r=np.random.default_rng(seed); ml=Sequential(Flatten(),Dense(36,12,r),Activation('relu'),Dense(12,2,r))
        for m in (cnn,ml): train(m,x[:120],y[:120],Optimizer('adam',.01),epochs=35,batch=20,seed=seed)
        add(8,seed,'CNN vs MLP','held-out accuracy',np.mean(cnn.forward(x[120:]).argmax(1)==y[120:]),np.mean(ml.forward(x[120:]).argmax(1)==y[120:]),'dense MLP (different parameter counts)')
    return rows


def compare(a,b):
    assert len(a)==len(b)==63
    for x,y in zip(a,b):
        assert x.keys()==y.keys()
        for k,v in x.items():
            if isinstance(v,float): np.testing.assert_allclose(v,y[k],atol=1e-9,rtol=0)
            else: assert v==y[k]


def main():
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype']='none'
    a,b=run(),run(); compare(a,b); summary=[]
    Path('results').mkdir(exist_ok=True); Path('figures').mkdir(exist_ok=True)
    for exp,title in zip(sorted({r['experiment'] for r in a}),TITLES):
        rows=[r for r in a if r['experiment']==exp]; group=[]
        for c in dict.fromkeys(r['configuration'] for r in rows):
            rr=[r for r in rows if r['configuration']==c]; item={k:rr[0][k] for k in ('experiment','configuration','metric','reference_kind')}; item['n']=len(rr)
            for k in ('ours','reference'):
                vals=[r[k] for r in rr]; item[k+'_mean']=float(np.mean(vals)); item[k+'_sd']=float(np.std(vals,ddof=1))
            group.append(item)
        summary.extend(group); fig,ax=plt.subplots(figsize=(8,4)); pos=np.arange(len(group))
        for i,k in enumerate(('ours','reference')): ax.bar(pos+(i-.5)*.36,[r[k+'_mean'] for r in group],.36,yerr=[r[k+'_sd'] for r in group],label=k,capsize=3)
        ax.set_xticks(pos,[r['configuration'] for r in group],rotation=22,ha='right'); ax.set_ylabel(group[0]['metric']); ax.set_title(exp+' — '+title+' (3 seeds)'); ax.legend(); fig.tight_layout(); fig.savefig('figures/'+exp+'.svg'); fig.savefig('figures/'+exp+'.png'); plt.close(fig)
    payload=dict(project='AIR-004',seeds=SEEDS,environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,platform=platform.system()),reproducibility=dict(runs=2,atol=1e-9,rtol=0,status='pass'),records=a,summary=summary)
    Path('results/benchmarks.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(len(a),'records;',len(summary),'configurations; two runs matched')

if __name__=='__main__': main()
