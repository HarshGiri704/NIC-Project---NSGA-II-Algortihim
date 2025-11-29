import numpy as np

def normalize(F, ideal_point, nadir_point):
    """
    Normalize objective values F (shape: [N, 2]), where F can be array-like or np.array
    Returns the normalized front values (shape: [N, 2])
    ideal_point: np.array([min_time, min_profit])
    nadir_point: np.array([max_time, max_profit])
    Assumes minimization of both objectives.
    """
    F = np.array(F)
    norm = (F - ideal_point) / (nadir_point - ideal_point + 1e-9)
    return norm
