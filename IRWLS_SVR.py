import numpy as _np
from kernels import RBF

class IRWLS_SVR():
    """Implementation of the Iterative Re-Weighted Least Sqaure procedure for
    fitting a Support Vector Regressor.


    See [Perez-Cruz et al., 2000]
    (http://ieeexplore.ieee.org/document/7075361/)
    ([pdf](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7075361)).
    """
    def __init__(self, C1 = 1.0, C2 = 1.0, epsilon = 0.1, gamma = 1.0):
        """Construct a new regressor

        Args:
          C1: Penalty parameter C of the error term.
          C2: Penalty parameter of the error in the derivaties.
          epsilon: insensitive region of the error function.
          gamma: Factor for the exponential in the RBF kernel. Defaults to 1.0.
        """
        self.C1 = C1
        self.C2 = C2
        self.epsilon = epsilon
        self.kernel = RBF(gamma = gamma)
        self._is_fitted = False

    def _lagrange(self, w2, e, e_star, d, d_star):
        return 0.5*w2 + self.C1*_np.sum(_np.maximum(0, e) + _np.maximum(0, e_star)) +\
            self.C2*_np.sum(_np.maximum(0, d) + _np.maximum(0, d_star))

    def fit(self, x_train, y_train, x_prime_train = None, y_prime_train = None,
        max_iter = 1000, plot_matrices = False):
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

        y_prime_train = y_prime_train.flatten()

        K = self.kernel(x_train, x_train)
        K_prime = self.kernel(x_prime_train, x_train, dx = -1, dy = 0)
        G = K_prime.T
        J = self.kernel(x_prime_train, x_prime_train, dx = 0, dy = 0)

        # In a first step the full matrix without the diagonal terms D_a+a* is
        # constructed. This would need to be changed when dealing with large
        # amounts of data points.
        full_matrix = _np.zeros((len(x_train) + len(x_prime_train) + 1,
                                len(x_train) + len(x_prime_train) + 1))
        full_matrix[:len(x_train), :len(x_train)] = K
        full_matrix[:len(x_train), len(x_train):-1] = G
        full_matrix[len(x_train):-1, :len(x_train)] = K_prime
        full_matrix[len(x_train):-1, len(x_train):-1] = J
        full_matrix[:len(x_train), -1] = 1.0
        full_matrix[-1, :len(x_train)] = 1.0

        a = _np.zeros(len(x_train))
        a[0::2] = self.C1
        a_star = _np.zeros(len(x_train))
        a_star[1::2] = self.C1
        s = _np.zeros(len(x_prime_train))
        s[0::2] = self.C2
        s_star = _np.zeros(len(x_prime_train))
        s_star[1::2] = self.C2

        beta_gamma_s = _np.zeros(len(x_train) + len(x_prime_train))
        beta_gamma = _np.zeros(len(x_train) + len(x_prime_train))

        active_a = _np.ones(len(x_train), dtype = bool)
        active_s = _np.ones(len(x_prime_train), dtype = bool)

        iter_counter = 0
        converged = False
        self.Ls = []
        self.coeffs = []

        while not converged:
            eta = 1.0
            N_a = _np.sum(active_a)
            N_s = _np.sum(active_s)
            # Build vector of all active coefficients (a, s and b) for masking
            active_coeffs = _np.concatenate([active_a, active_s,
                _np.ones(1, dtype = bool)])

            # Setup reduced matrix
            mat = full_matrix[_np.logical_and.outer(active_coeffs,
                active_coeffs)].reshape((N_a + N_s + 1, N_a + N_s + 1))
            mat[:N_a, :N_a] += _np.diag(1.0/(a + a_star)[active_a])
            mat[N_a:-1, N_a:-1] +=  _np.diag(1.0/(s + s_star)[active_s])
            if N_a == 0:
                mat[-1, -1] = 1.0

            if plot_matrices:
                import matplotlib.pyplot as _plt
                _plt.figure()
                _plt.pcolormesh(mat)
                _plt.colorbar()
                _plt.ylim(len(mat),0)
            # Setup corresponding target vector
            target = _np.concatenate([
                (a - a_star)[active_a]/(a + a_star)[active_a] * self.epsilon +
                y_train[active_a],
                (s - s_star)[active_s]/(s + s_star)[active_s]*self.epsilon +
                y_prime_train[active_s], _np.zeros(1)])

            beta_gamma_b = _np.linalg.solve(mat, target)
            beta_gamma_s[active_coeffs[:-1]] = beta_gamma_b[:-1]
            beta_gamma_s[_np.logical_not(active_coeffs[:-1])] = 0.0
            b_s = beta_gamma_b[-1]


            if iter_counter > 0:
                while True:
                    beta_gamma = beta_gamma_old.copy() + eta*(beta_gamma_s.copy() - beta_gamma_old.copy())
                    b = b_old.copy() + eta*(b_s.copy() - b_old.copy())
                    # Calculate errors
                    e = K.dot(beta_gamma[:len(x_train)]) + \
                        G.dot(beta_gamma[len(x_train):]) + b - y_train - self.epsilon
                    e_star = y_train - K.dot(beta_gamma[:len(x_train)]) - \
                        G.dot(beta_gamma[len(x_train):]) - b - self.epsilon
                    d = K_prime.dot(beta_gamma[:len(x_train)]) + \
                        J.dot(beta_gamma[len(x_train):]) - y_prime_train - self.epsilon
                    d_star = y_prime_train - K_prime.dot(beta_gamma[:len(x_train)]) - \
                        J.dot(beta_gamma[len(x_train):]) - self.epsilon
                    if self._lagrange(beta_gamma.T.dot(full_matrix[:-1, :-1]).dot(beta_gamma),
                        e, e_star, d, d_star) > self.Ls[-1]:
                        eta = eta*0.9
                    else:
                        # Was used for debugging, not sure if it can be remove
                        # so commented for now:
                        # print("Inner Loop converged with eta = {}".format(eta))
                        break

            else:
                beta_gamma = beta_gamma_s.copy()
                b = b_s.copy()
                # Calculate errors
                e = K.dot(beta_gamma[:len(x_train)]) + \
                    G.dot(beta_gamma[len(x_train):]) + b - y_train - self.epsilon
                e_star = y_train - K.dot(beta_gamma[:len(x_train)]) - \
                    G.dot(beta_gamma[len(x_train):]) - b - self.epsilon
                d = K_prime.dot(beta_gamma[:len(x_train)]) + \
                    J.dot(beta_gamma[len(x_train):]) - y_prime_train - self.epsilon
                d_star = y_prime_train - K_prime.dot(beta_gamma[:len(x_train)]) - \
                    J.dot(beta_gamma[len(x_train):]) - self.epsilon

            # a and s are restricted by a maximal error
            a = _np.minimum(_np.maximum(0, self.C1/e), 1e8)
            a_star = _np.minimum(_np.maximum(0, self.C1/e_star), 1e8)
            s = _np.minimum(_np.maximum(0, self.C2/d), 1e8)
            s_star = _np.minimum(_np.maximum(0, self.C2/d_star), 1e8)

            # Calculate active coefficients for next step
            active_a[_np.logical_and(active_a,
                _np.logical_and(a == 0.0, a_star == 0.0))] = False
            active_a[_np.logical_and(_np.logical_not(active_a),
                _np.logical_or(a != 0.0, a_star != 0.0))] = True
            active_s[_np.logical_and(active_s,
                _np.logical_and(s == 0.0, s_star == 0.0))] = False
            active_s[_np.logical_and(_np.logical_not(active_s),
                _np.logical_or(s != 0.0, s_star != 0.0))] = True

            self.Ls.append(self._lagrange(beta_gamma.T.dot(full_matrix[:-1, :-1]).dot(beta_gamma),
                e, e_star, d, d_star))
            self.coeffs.append(beta_gamma)    

            # Check for convergence
            if iter_counter > 0:
                if (_np.linalg.norm(beta_gamma - beta_gamma_old) < 1e-10 and
                    _np.abs(b - b_old) < 1e-10):
                    print("Converged after {} iterations".format(iter_counter))
                    converged = True

            if iter_counter >= max_iter:
                print("Maximum iterations ({}) reached!".format(max_iter))
                if iter_counter > 0:
                    print("Final Deltas: beta_gammma: {: 14.12f}, b: {: 14.12f}".format(
                        _np.linalg.norm(beta_gamma - beta_gamma_old), _np.abs(b - b_old)))
                converged = True

            # Save old values for convergence testing
            beta_gamma_old = beta_gamma.copy()
            b_old = b.copy()
            iter_counter += 1

        self.x_train = x_train[active_a]
        self.x_prime_train = x_prime_train[active_s]
        self.beta_gamma = beta_gamma[_np.concatenate([active_a, active_s])]
        self.intercept = b

        self._is_fitted = True

    def predict(self, x_test):
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
        return self.kernel(x_test, self.x_train).dot(
            self.beta_gamma[:len(self.x_train)]) + \
            self.kernel(x_test, self.x_prime_train, dx = 0, dy = -1).dot(
            self.beta_gamma[len(self.x_train):]) + self.intercept

    def predict_derivative(self, x_test):
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
        return self.kernel(x_test, self.x_train, dx = -1, dy = 0).dot(
            self.beta_gamma[:len(self.x_train)]) + \
            self.kernel(x_test, self.x_prime_train, dx = 0, dy = 0).dot(
            self.beta_gamma[len(self.x_train):])
