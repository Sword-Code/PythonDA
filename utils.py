import numpy as np

def mean_and_base(ensemble, weights=None):        
    ensemble[...,0,:]=np.average(ensemble,axis=-2,weights=weights)
    ensemble[...,1:,:]-=ensemble[...,0:1,:]        
    return ensemble

def mean_std(ensemble, weights=None, mean=None):
    if mean is None:
        mean=np.average(ensemble,axis=-2,weights=weights)
    else:
        mean=mean.copy()    
    mean=mean[...,None,:]
    anomalies=ensemble-mean
    std=np.sqrt(np.average(anomalies**2,axis=-2,weights=weights))
    mean=mean[...,0,:]
    return mean, std

def transpose(matrices):
    return matrices.transpose(list(range(len(matrices.shape)-2))+[-1,-2])

def ortmatrix(matrices,start):
    shape=matrices.shape    
    matrices[...,start:,:]=np.random.normal(size= shape[:-2]+(shape[-2]-start, shape[-1]))
    #print(matrices)
    matrices[...,start:,:] /= np.linalg.norm(matrices[...,start:,:], axis=-1, keepdims=True)
    #print(matrices)
    for k in range(1,shape[-2]):
        i=np.amax([k,start])
        matrices[...,i:,:] -= np.matmul(np.matmul(matrices[...,i:,:],matrices[...,k-1,:, None]),matrices[...,k-1:k,:])
        #print(matrices)
        matrices[...,i:,:]/=np.linalg.norm(matrices[...,i:,:], axis=-1, keepdims=True)
        #print(matrices)
        
    return matrices 
