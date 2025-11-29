import numpy as np

class Hypervolume:
    """
    Computes the hypervolume of a set of two-dimensional points
    w.r.t. a given reference point (for bi-objective minimization).
    """
    def __init__(self, ref_point):
        self.ref_point = ref_point

    def calc(self, front):
        """
        front: np.array of shape (N, 2) for bi-objective minimization
        Returns the hypervolume.
        """
        # Sort front by first objective (time)
        front = front[front[:,0].argsort()]
        hv = 0.0
        for i in range(len(front)):
            x, y = front[i]
            if i == len(front)-1:
                x_next = self.ref_point[0]
            else:
                x_next = front[i+1,0]
            width = self.ref_point[0] - x
            height = self.ref_point[1] - y
            if width > 0 and height > 0:
                hv += width * height
        return hv
