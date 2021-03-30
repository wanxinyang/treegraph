import numpy as np

# copied from https://stackoverflow.com/a/50728570/1414831, thank you to Hans Musgrave

def t(p, q, r):
    x = p-q
    return np.dot(r-q, x)/np.dot(x, x)

def d(p, q, rs):
    x = p - q
    return np.linalg.norm(np.outer(np.dot(rs-q, x)/np.dot(x, x), x)+q-rs, axis=1)