import numpy as np

def generate(data, Np):
    """Reduces (NxD) data matrix from N to Np data points.

    Args:
        data: ndarray of shape [N, D]
        Np: number of data points in the coreset
    Returns:
        coreset: ndarray of shape [Np, D]
        weights: 1darray of shape [Np, 1]
    """
    N = data.shape[0]
    D = data.shape[1]

    # compute mean
    u = np.mean(data, axis=0)

    # compute proposal distribution
    q = np.linalg.norm(data - u, axis=1)**2
    sum = np.sum(q)
    q = 0.5 * (q/sum + 1.0/N)

    # get sample and fill coreset
    samples = np.random.choice(N, Np, p=q)
    coreset = data[samples]
    weights = 1.0 / (q[samples] * Np)
    
    return coreset, weights
