import numpy as _np

# ToDo: maybe rewritte so dx = dy = 0 is the default case. This would be more
# consistent with the definition of an derivative.
class RBF(object):

    def __init__(self, gamma = 1.0):
        self.gamma = gamma

    def __call__(self, x, y, dx = -1, dy = -1):
        exp_mat = _np.exp(-self.gamma *
            (_np.tile(_np.sum(x**2, axis = 1), (len(y), 1)).T +
            _np.tile(_np.sum(y**2, axis = 1), (len(x), 1)) - 2*x.dot(y.T)))
        if dx == dy:
            if dx == -1:
                return exp_mat
            else:
                return -2.0 * self.gamma * exp_mat * (2.0 * self.gamma * \
                    _np.subtract.outer(x[:, dy].T, y[:, dy])**2 - 1)
        elif dx == -1:
            return -2.0 * self.gamma * exp_mat * \
                _np.subtract.outer(x[:, dy].T, y[:, dy])
        elif dy == -1:
            return 2.0 * self.gamma * exp_mat * \
                _np.subtract.outer(x[:, dx].T, y[:, dx])
        else:
            return -4.0 * self.gamma**2 * exp_mat * \
                _np.subtract.outer(x[:, dx].T, y[:, dx]) * \
                _np.subtract.outer(x[:, dy].T, y[:, dy])
