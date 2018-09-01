import numpy as np

def decode(c):
    if c == 1:
        return 9
    elif c == 127 - 30:
        return 10
    elif 32 <= c + 30 <= 126:
        return c + 30
    else:
        return 0

def encode(c):
    if c == 9:
        return 1
    elif c == 10:
        return 127 - 30
    elif 32 <= c <= 126:
        return c - 30
    else:
        return 0

def sample(p, topn=98):
    p[np.argsort(p)[:-topn]] = 0
    p = p / np.sum(p)
    return np.random.choice(98, 1, p=p)[0]
