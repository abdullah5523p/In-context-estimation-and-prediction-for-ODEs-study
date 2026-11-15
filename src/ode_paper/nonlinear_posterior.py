"""Randomized importance quadrature for nonlinear uniform-box posteriors.

Independent Sobol scrambles, effective sample size, and successive resolutions
are checked. These diagnostics do not prove that all posterior modes were found.
"""
import hashlib, json, os
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.special import log_ndtr, logsumexp
from scipy.stats import qmc, truncnorm
from .config import fingerprint
from .families import SPECS, bounds, decode, solution

def _lm(a,b):
    u=np.where(a>0,log_ndtr(-a),log_ndtr(b))
    l=np.where(a>0,log_ndtr(-b),log_ndtr(a))
    return u+np.log(-np.expm1(l-u))

def _setup(point,family,q,initial,starts,seed):
    p=len(initial); scale=np.asarray(SPECS[family]['scale'])
    times=point[:,0].astype(float); y=point[:,1:].astype(float)
    def residual(z):
        return (solution(decode(z[None],family),times[None],family)[0]/scale-y).ravel()
    rng=np.random.default_rng(seed); zstarts=[np.asarray(initial,float)]
    if family=='stuart_landau':
        for w in (-.72,-.36,0.,.36,.72):
            z=zstarts[0].copy(); z[3]=w; zstarts.append(z)
    while len(zstarts)<starts: zstarts.append(rng.uniform(-.8,.8,p))
    modes=[]
    for z0 in zstarts:
        fit=least_squares(residual,z0,bounds=(-.8,.8),max_nfev=500,
                          ftol=1e-9,xtol=1e-9,gtol=1e-9)
        if not (np.isfinite(fit.fun).all() and np.isfinite(fit.jac).all()): continue
        prec=fit.jac.T@fit.jac/q**2+np.eye(p)
        cov=np.linalg.solve(prec,np.eye(p))
        center=np.clip(fit.x+cov@(-fit.jac.T@fit.fun/q**2-fit.x),-1.5,1.5)
        order=np.argsort(np.diag(cov)); undo=np.argsort(order)
        chol=np.linalg.cholesky(cov[np.ix_(order,order)])
        item=(center[order],chol,undo)
        # Compare centers in their original coordinates, before each proposal's
        # own Cholesky ordering. Otherwise distinct modes can be merged.
        if not any(np.linalg.norm(center-m[0][m[2]])<1e-3 for m in modes): modes.append(item)
    if not modes: raise RuntimeError('Nonlinear posterior mode searches failed')
    return modes,times,y,scale

def _draw(mode,power,seed):
    mu,L,undo=mode;p=len(mu);u=qmc.Sobol(p,scramble=True,seed=seed).random_base2(power)
    e=np.zeros_like(u);zo=np.zeros_like(u)
    for j in range(p):
        c=mu[j]+e[:,:j]@L[j,:j];a=(-.8-c)/L[j,j];b=(.8-c)/L[j,j]
        e[:,j]=truncnorm.ppf(np.clip(u[:,j],1e-15,1-1e-15),a,b)
        zo[:,j]=c+L[j,j]*e[:,j]
    z=np.empty_like(zo);z[:,np.argsort(undo)]=zo
    return z

def _logq(z,mode):
    mu,L,undo=mode;zo=z[:,np.argsort(undo)];e=np.zeros_like(zo);v=np.zeros(len(z))
    for j in range(zo.shape[1]):
        c=mu[j]+e[:,:j]@L[j,:j];a=(-.8-c)/L[j,j];b=(.8-c)/L[j,j]
        e[:,j]=(zo[:,j]-c)/L[j,j]
        v+=-.5*e[:,j]**2-.5*np.log(2*np.pi)-np.log(L[j,j])-_lm(a,b)
    return v

def posterior_one(point,family,q,initial,seed,min_power=10,max_power=15,
                  replicates=4,tolerance=.03,min_ess=128,starts=6):
    if family not in ('logistic','birth_death','stuart_landau'):
        raise ValueError('Choose one of the three nonlinear ODE families')
    if not (q>0 and max_power>=min_power+2 and min_power>=2 and replicates>=2
            and tolerance>0 and min_ess>0 and starts>=1):
        raise ValueError('Require positive settings and max_power >= min_power + 2')
    modes,times,y,scale=_setup(point,family,q,initial,starts,seed)
    prev=None; prev_var=None; passes=0
    for power in range(min_power,max_power+1):
        reps=[]
        for rep in range(replicates):
            z=np.concatenate([_draw(m,power,seed+10007*rep+1009*k) for k,m in enumerate(modes)])
            # A defensive uniform-prior component gives the quadrature direct
            # access to the entire box, including regions absent from the fits.
            u=qmc.Sobol(len(initial),scramble=True,
                        seed=seed+10007*rep+1009*len(modes)).random_base2(power)
            z=np.concatenate((z,1.6*u-.8))
            lq=np.stack([_logq(z,m) for m in modes],axis=1)
            uniform_logq=np.full((len(z),1),-len(initial)*np.log(1.6))
            lq=logsumexp(np.concatenate((lq,uniform_logq),axis=1),axis=1)-np.log(len(modes)+1)
            # Bound temporary memory for large n and high quadrature resolutions.
            lw=np.empty(len(z))
            for start in range(0,len(z),2048):
                stop=min(start+2048,len(z)); block=z[start:stop]
                pred=solution(decode(block,family),np.broadcast_to(times,(len(block),len(times))),family)/scale
                lw[start:stop]=-np.sum((pred-y[None,:,:])**2,axis=(1,2))/(2*q*q)-lq[start:stop]
            w=np.exp(lw-logsumexp(lw));mean=w@z
            if not (np.isfinite(w).all() and np.isfinite(z).all()):
                raise FloatingPointError('Non-finite posterior quadrature weights or samples')
            cov=((z-mean)*w[:,None]).T@(z-mean)
            features=z
            if family=='birth_death':
                theta=decode(z,family)
                features=np.column_stack((z,theta[:,0]-theta[:,1],theta[:,2]))
            fmean=w@features
            fvar=w@((features-fmean)**2)
            reps.append((mean,cov,1/np.sum(w*w),fmean,fvar,w.max()))
        ms=np.array([x[0] for x in reps]);m=ms.mean(0)
        cov=np.mean([x[1]+np.outer(x[0]-m,x[0]-m) for x in reps],axis=0)
        se=ms.std(0,ddof=1)/np.sqrt(replicates)
        fmeans=np.array([x[3] for x in reps]); fm=fmeans.mean(0)
        fvar=np.mean([x[4]+(x[3]-fm)**2 for x in reps],axis=0)
        fsd=np.sqrt(np.maximum(fvar,1e-24))
        fse=fmeans.std(0,ddof=1)/np.sqrt(replicates)
        shift=np.inf if prev is None else float(np.max(np.abs(fm-prev)/fsd))
        variance_shift=np.inf if prev_var is None else float(np.max(np.abs(fvar-prev_var)/np.maximum(fvar,1e-24)))
        ess=min(x[2] for x in reps); rel=float(np.max(fse/fsd))
        max_weight=max(x[5] for x in reps)
        good=(rel<=tolerance and shift<=3*tolerance and ess>=min_ess
              and variance_shift<=max(.1,10*tolerance) and max_weight<=.02)
        passes=passes+1 if good else 0
        ok=passes>=2
        if ok or power==max_power:
            return dict(mean=m,covariance=cov,integration_se=se,min_ess=ess,power=power,
                        converged=ok,relative_se=rel,relative_shift=shift,mode_count=len(modes),
                        variance_shift=variance_shift,max_weight=max_weight,
                        monitored_mean=fm,monitored_variance=fvar,monitored_se=fse,
                        stable_refinements=passes)
        prev=fm;prev_var=fvar

def cached_posterior(root,points,family,q,seed,n,initials,**opts):
    settings=dict(min_power=10,max_power=15,replicates=4,tolerance=.03,min_ess=128,starts=6);settings.update(opts)
    ident=dict(family=family,q=q,seed=seed,n=n,settings=settings,
        points=hashlib.sha256(points.tobytes()).hexdigest(),initials=hashlib.sha256(initials.tobytes()).hexdigest(),
        code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        families=hashlib.sha256(Path(__file__).with_name('families.py').read_bytes()).hexdigest())
    folder=Path(root)/'statistical_comparison_cache'/f'{family}_posterior'/fingerprint(ident)
    folder.mkdir(parents=True,exist_ok=True);out=[];worst=0.
    manifest=folder/'settings.json'
    if not manifest.exists(): manifest.write_text(json.dumps(ident,indent=2))
    for i,pt in enumerate(points):
        path=folder/f'example_{i:04d}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as d:r={k:d[k].copy() for k in d.files}
        else:
            r=posterior_one(pt,family,q,initials[i],seed+1009*n+7919*i,**settings)
            tmp=folder/f'example_{i:04d}.building.npz';np.savez_compressed(tmp,**r);os.replace(tmp,path)
        if not bool(r['converged']):
            raise RuntimeError(f'{family} posterior failed convergence for example {i}, n={n}: '
                f"ESS={float(r['min_ess']):.1f}, relative SE={float(r['relative_se']):.3g}, "
                f"successive shift={float(r['relative_shift']):.3g}, "
                f"variance shift={float(r['variance_shift']):.3g}, max weight={float(r['max_weight']):.3g}; "
                f"increase max_power or starts. Cached: {path}")
        out.append(r['mean']);worst=max(worst,float(r['relative_se']))
        if (i+1)%32==0 or i+1==len(points):
            print(f'  Posterior {family} seed={seed} n={n}: {i+1}/{len(points)}, max rel. integration SE={worst:.3g}',flush=True)
    return np.asarray(out)
