import numpy as _np

def RBF_Kernel(x, y, nx = -1, ny = -1, gamma = 1.0):
    exp_mat = _np.exp(-gamma * (_np.tile(_np.sum(x**2, axis = 1), (len(y), 1)).T +
        _np.tile(_np.sum(y**2, axis = 1), (len(x), 1)) - 2*x.dot(y.T)))
    if nx == ny:
        if nx == -1:
            return exp_mat
        else:
            return -2.0 * gamma * exp_mat * \
                (2.0 * gamma * _np.subtract.outer(x[:,ny].T, y[:,ny])**2 - 1)
    elif nx == -1:
        return -2.0 * gamma * exp_mat * _np.subtract.outer(x[:,ny].T, y[:,ny])
    elif ny == -1:
        return 2.0 * gamma * exp_mat * _np.subtract.outer(x[:,nx].T, y[:,nx])
    else:
        return -4.0 * gamma**2 * exp_mat * \
            _np.subtract.outer(x[:,nx].T, y[:,nx]) * \
            _np.subtract.outer(x[:,ny].T, y[:,ny])


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
        self.gamma = gamma
        self.method = method
        self._is_fitted = False

    def fit(self, x_train, y_train, x_prime_train, y_prime_train,
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
        self.x_train = x_train
        self.x_prime_train = x_prime_train
        self.dim = x_train.shape[1]

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
                mat[0,0]=1

        # Populate matrix. Would be significantly simpler under the assumption
        # that x_train == x_prime_train.
        for nx in range(-1, self.dim):
            for ny in range(nx, self.dim):
                # find pairs of indices determining the position of certain
                # blocks within the matrix
                if nx == -1 and ny == -1:
                    ind1 = [0, len(x_train)]
                    ind2 = [0, len(x_train)]
                elif nx == -1:
                    ind1 = [len(x_train) + (ny)*len(x_prime_train),
                            len(x_train) + (ny + 1)*len(x_prime_train)]
                    ind2 = [0, len(x_train)]
                elif ny == -1:
                    ind1 = [0, len(x_train)]
                    ind2 = [len(x_train) + (nx)*len(x_prime_train),
                            len(x_train) + (nx + 1)*len(x_prime_train)]
                else:
                    ind1 = [len(x_train) + (ny)*len(x_prime_train),
                            len(x_train) + (ny + 1)*len(x_prime_train)]
                    ind2 = [len(x_train) + (nx)*len(x_prime_train),
                            len(x_train) + (nx + 1)*len(x_prime_train)]
                if self.method == 1:
                    ind1 = [i + 1 for i in ind1]
                    ind2 = [i + 1 for i in ind2]

                if nx == ny:
                    if nx == -1:
                        if self.method == 0:
                            mat[ind1[0]:ind1[1], ind2[0]:ind2[1]] = RBF_Kernel(
                                x_train, x_train, gamma = self.gamma, nx = nx,
                                ny = ny) + 1 + _np.eye(len(x_train))/self.C1
                        else:
                            mat[ind1[0]:ind1[1], ind2[0]:ind2[1]] = RBF_Kernel(
                                x_train, x_train, gamma = self.gamma, nx = nx,
                                ny = ny) + _np.eye(len(x_train))/self.C1
                    else:
                        mat[ind1[0]:ind1[1], ind2[0]:ind2[1]] = RBF_Kernel(
                            x_prime_train, x_prime_train, gamma = self.gamma,
                            nx = nx, ny = ny) + \
                            _np.eye(len(x_prime_train))/self.C2
                else:
                    mat[ind1[0]:ind1[1], ind2[0]:ind2[1]] = RBF_Kernel(
                        x_prime_train, x_train, gamma = self.gamma, nx = nx,
                        ny = ny)
                    mat[ind2[0]:ind2[1], ind1[0]:ind1[1]] = RBF_Kernel(
                        x_prime_train, x_train, gamma = self.gamma, nx = nx,
                        ny = ny).T

        # ToDo: Implement partition scheme for inverting the matrix
        matinv = _np.linalg.inv(mat)

        if self.method == 0:
            a_b = matinv.dot(_np.concatenate(
                [y_train] + [y_prime_train[:,i] for i in range(self.dim)]))
            self.a = a_b[0:len(x_train)]
            self.b = a_b[len(x_train):].reshape((self.dim, -1)).T
            self.intercept = sum(self.a)
        elif self.method == 1:
            a_b = matinv.dot(_np.concatenate([_np.zeros(1), y_train] +
                [y_prime_train[:,i] for i in range(self.dim)]))
            self.a = a_b[1:len(x_train)+1]
            self.b = a_b[len(x_train)+1:].reshape((self.dim, -1)).T
            self.intercept = a_b[0]

        self._is_fitted = True
        if plot_matrices:
            self._plot_matrices(mat, matinv)

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
        return self.a.dot(RBF_Kernel(self.x_train, x, gamma = self.gamma)) + \
            sum([ self.b[:,i].dot(RBF_Kernel(
            self.x_prime_train, x, gamma = self.gamma, ny = i))
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
        ret_mat = _np.zeros((len(x), self.dim))
        for i in range(self.dim):
            ret_mat[:,i] = self.a.dot(RBF_Kernel(
                self.x_train, x, gamma = self.gamma, nx = i)) + \
                sum([self.b[:,j].dot(RBF_Kernel(
                self.x_prime_train, x, gamma = self.gamma, nx = i, ny = j))
                for j in range(self.dim)])
        if not self._is_fitted:
            raise ValueError("Instance is not fitted yet")
        return ret_mat

    def _plot_matrices(self, mat, matinv):
        """ Used for debugging"""
        import matplotlib.pyplot as _plt
        fig, (ax0, ax1) = _plt.subplots(ncols=2, figsize = (12,5))
        p0 = ax0.pcolormesh(mat)
        ax0.set_ylim(len(mat),0)
        fig.colorbar(p0, ax = ax0)
        p1 = ax1.pcolormesh(matinv)
        ax1.set_ylim(len(mat),0)
        fig.colorbar(p1, ax = ax1)
