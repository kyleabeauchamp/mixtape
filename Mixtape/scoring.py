import numpy as np

def single_log_likelihood(ass, T, ni):
    ll = np.log(T[ass[:-1], ass[1:]]).sum()
    ll -= (ni * np.log(ni)).sum()
    ll += np.log(ni[ass[0]])
    return ll



def log_likelihood(sequences, T):    
    ni = np.bincount(np.concatenate(sequences))
    ll = 0.
    for k, a in enumerate(sequences):
        ll += single_log_likelihood(a, T, ni)
    return ll

def assign(X, cutoff):
    return (X > cutoff).astype('int')
