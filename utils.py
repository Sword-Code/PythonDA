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
        
def log_likelihood(obs, state, A1, mean_and_base):
    lndetA1=np.log(np.linalg.det(A1))
    if obs==[]:
        return np.zeros(state.shape[:-2])
    Hstate=obs.H(state)
    Hstate=mean_and_base(Hstate)
    Hstate[...,0,:]=obs.misfit(Hstate[...,0,:])
    
    if Hstate.shape[-2]<obs.obs.shape[-1]:
        sqrtR1HL=obs.sqrtR1(Hstate)
        HLTR1HL=np.matmul(sqrtR1HL,transpose(sqrtR1HL[...,1:,:]))
        temp=A1+HLTR1HL[...,1:,:]
        eigenvalues, eigenvectors = np.linalg.eigh(temp)
        sqrteig=np.sqrt(eigenvalues)
        temp=np.matmul(HLTR1HL[...,:1,:],eigenvectors)/sqrteig[...,None,:]
        score=(sqrtR1HL[...,0,:]**2).sum(-1) - (temp**2).sum((-1,-2)) - lndetA1 + np.log(obs.std.prod(-1)**2) + np.log(sqrteig.prod(-1))*2
        
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(A1)
        temp=np.matmul(transpose(eigenvectors/np.sqrt(eigenvalues[...,None,:])),Hstate[...,1:,:])
        PL=np.eye(obs.std.shape[-1])*obs.std[...,None,:]**2+np.matmul(transpose(temp),temp)
        eigenvalues, eigenvectors = np.linalg.eigh(PL)
        temp=np.matmul(transpose(eigenvectors/np.sqrt(eigenvalues[...,None,:])),Hstate[...,0,:, None])
        score=(temp**2).sum((-1,-2))+np.log(np.sqrt(eigenvalues)).sum(-1)*2
        
    return score
