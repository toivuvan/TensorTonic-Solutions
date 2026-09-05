import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    # Write code here
    result = []
    x = np.asarray(x)
    for a in x:
        if a.ndim == 1:
            i = np.exp(a - np.max(a)) / np.sum(np.exp(a - np.max(a)))
            result.append(i)
        else:
            return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
            
    return np.asarray(result)          
        
        
    