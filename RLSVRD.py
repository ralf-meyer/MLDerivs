import numpy as _np
from kernels import RBF

class RLSVRD(object):
    """Implementation of the regularized least squares support vector
    regression algorithm for the simulatneous learning of a function and
    its derivatives using the RBF Kernel.


    See [Jayadeva et al., 2008]
    (https://www.sciencedirect.com/science/article/pii/S0020025508001291)
    ([pdf](http://isiarticles.com/bundles/Article/pre/pdf/24941.pdf)).
    """

    def __init__(self, C1 = 1.0, C2 = 1.0, gamma = 1.0, method = 1):
        """Construct a new regressor

        Args:
          C1: Penalty parameter C of the error term.
          C2: Penalty parameter of the error in the derivaties.
          gamma: Factor for the exponential in the RBF kernel. Defaults to 1.0.
          method: Determines whether the intercept should be part of the
            regularization (as in the original publication) (method = 0) or
            not (method = 1). Defaults to 1.
        """
        self.C1 = C1
        self.C2 = C2
        self.kernel = RBF(gamma = gamma)
        self.method = method
        self._is_fitted = False

    def fit(self, x_train, y_train, x_prime_train = None, y_prime_train = None,
            plot_matrices = False):
        """Fit the model to given training data
        Parameters:
          x_train: shape (n_samples, n_features)
            Trainings vectors.
          y_train: shape (n_samples)
            Values of the target function at the trainings vectors x_train.
          x_prime_train: shape(n_derivative_samples, n_features)
            Trainings vectors for the derivatives. Seperate input as the
            algorithm allows these to be different from x_train.
          y_prime_train: shape(n_derivative_samples, n_features)
            Contains the derivatives of the target function at the points
            x_prime_train with respect to all dimensions (features)
        Returns:
            self
        """
        self.dim = x_train.shape[1]
        if x_prime_train is None:
            x_prime_train = _np.zeros((0, self.dim))
        if y_prime_train is None:
            y_prime_train = _np.zeros((0, self.dim))

        self.x_train = x_train
        self.x_prime_train = x_prime_train

        if self.method == 0:
            mat = _np.zeros((len(x_train) + len(x_prime_train)*self.dim,
                            len(x_train) + len(x_prime_train)*self.dim))
        elif self.method == 1:
            mat = _np.zeros((1 + len(x_train) + len(x_prime_train)*self.dim,
                            1 + len(x_train) + len(x_prime_train)*self.dim))
            mat[1:len(x_train) + 1, 0] = 1
            mat[0, 1:len(x_train) + 1] = 1
            # avoid singular matrix when training with derivatives only
            if len(x_train) == 0:
                mat[-1, -1] = 1
        elif self.method == 2:
            mat = _np.zeros((1 + len(x_train) + len(x_prime_train)*self.dim,
                            1 + len(x_train) + len(x_prime_train)*self.dim))
            mat[0:len(x_train), -1] = 1
            mat[-1, 0:len(x_train)] = 1
            # avoid singular matrix when training with derivatives only
            if len(x_train) == 0:
                mat[-1, -1] = 1

        # Populate matrix. Would be significantly simpler under the assumption
        # that x_train == x_prime_train.
        for dx in range(-1, self.dim):
            for dy in range(dx, self.dim):
                # find pairs of indices determining the position of certain
                # blocks within the matrix
                if dx == -1 and dy == -1:
                    ind1 = [0, len(x_train)]
                    ind2 = [0, len(x_train)]
                elif dx == -1:
                    ind1 = [len(x_train) + (dy)*len(x_prime_train),
                            len(x_train) + (dy + 1)*len(x_prime_train)]
                    ind2 = [0, len(x_train)]
                elif dy == -1:
                    ind1 = [0, len(x_train)]
                    ind2 = [len(x_train) + (dx)*len(x_prime_train),
                            len(x_train) + (dx + 1)*len(x_prime_train)]
                else:
                    ind1 = [len(x_train) + (dy)*len(x_prime_train),
                            len(x_train) + (dy + 1)*len(x_prime_train)]
                    ind2 = [len(x_train) + (dx)*len(x_prime_train),
                            len(x_train) + (dx + 1)*len(x_prime_train)]
                if self.method == 1:
                    ind1 = [i + 1 for i in ind1]
                    ind2 = [i + 1 for i in ind2]

                if dx == dy:
                    if dx == -1:
                        if self.method == 0:
                            mat[ind1[0]:ind1[1], ind2[0]:ind2[1]] = self.kernel(
                                x_train, x_train, dx = dx, dy = dy) + 1 + \
                                _np.eye(len(x_train))/self.C1
                        else:
                            mat[ind1[0]:ind1[1], ind2[0]:ind2[1]] = self.kernel(
                                x_train, x_train, dx = dx, dy = dy) + \
                                _np.eye(len(x_train))/self.C1
                    else:
                        mat[ind1[0]:ind1[1], ind2[0]:ind2[1]] = self.kernel(
                            x_prime_train, x_prime_train, dx = dx, dy = dy) + \
                            _np.eye(len(x_prime_train))/self.C2
                else:
                    mat[ind1[0]:ind1[1], ind2[0]:ind2[1]] = self.kernel(
                        x_prime_train, x_train, dx = dx, dy = dy)
                    mat[ind2[0]:ind2[1], ind1[0]:ind1[1]] = self.kernel(
                        x_prime_train, x_train, dx = dx, dy = dy).T

        # ToDo: Implement partition scheme for inverting the matrix
        # For now numpy.linalg.solve is used
        #matinv = _np.linalg.inv(mat)

        if self.method == 0:
            a_b = _np.linalg.solve(mat, _np.concatenate(
                [y_train] + [y_prime_train[:,i] for i in range(self.dim)]))
            self.a = a_b[0:len(x_train)]
            self.b = a_b[len(x_train):].reshape((self.dim, -1)).T
            self.intercept = sum(self.a)
        elif self.method == 1:
            a_b = _np.linalg.solve(mat, _np.concatenate([_np.zeros(1),
                y_train] + [y_prime_train[:,i] for i in range(self.dim)]))
            self.a = a_b[1:len(x_train)+1]
            self.b = a_b[len(x_train)+1:].reshape((self.dim, -1)).T
            self.intercept = a_b[0]
        elif self.method == 2:
            a_b = _np.linalg.solve(mat, _np.concatenate([y_train] +
                [y_prime_train[:,i] for i in range(self.dim)] + [_np.zeros(1)]))
            self.a = a_b[0:len(x_train)]
            self.b = a_b[len(x_train):-1].reshape((self.dim, -1)).T
            self.intercept = a_b[-1]

        self._is_fitted = True
        if plot_matrices:
            self._plot_matrices(mat)

    def predict(self, x):
        """Predict the values of a fitted model for new feature vectors
        Parameters:
          x: shape (n_samples, n_features)
            Feature vectors on which the model should be evaluated
        Returns:
          y_pred: shape(n_samples,)
            Predictions of the model at the supplied feature vectors
        """
        if not self._is_fitted:
            raise ValueError("Instance is not fitted yet")
        return self.a.dot(self.kernel(self.x_train, x)) + \
            sum([ self.b[:,i].dot(self.kernel(self.x_prime_train, x, dy = i))
            for i in range(self.dim)]) + self.intercept

    def predict_derivative(self, x):
        """Predict the derivatives of a fitted model for new feature vectors
        Parameters:
          x: shape (n_samples, n_features)
            Feature vectors on which the model derivatives should be evaluated
        Returns:
          y_pred: shape(n_samples, n_features)
            Predictions of the model derivatives at the supplied feature vectors
        """
        if not self._is_fitted:
            raise ValueError("Instance is not fitted yet")
        ret_mat = _np.zeros((len(x), self.dim))
        for i in range(self.dim):
            ret_mat[:,i] = self.a.dot(self.kernel(
                self.x_train, x, dx = i)) + sum([self.b[:,j].dot(self.kernel(
                self.x_prime_train, x, dx = i, dy = j))
                for j in range(self.dim)])
        return ret_mat

    def _plot_matrices(self, mat):
        """ Used for debugging"""
        import matplotlib.pyplot as _plt
        fig, (ax0, ax1) = _plt.subplots(ncols=2, figsize = (12,5))
        p0 = ax0.pcolormesh(mat)
        ax0.set_ylim(len(mat),0)
        fig.colorbar(p0, ax = ax0)
        p1 = ax1.pcolormesh(np.linalg.inv(mat))
        ax1.set_ylim(len(mat),0)
        fig.colorbar(p1, ax = ax1)
