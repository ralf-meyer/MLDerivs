import numpy as _np
from kernels import RBF

class IRWLS_SVR():
    """Implementation of the Iterative Re-Weighted Least Sqaure procedure for
    fitting a Support Vector Regressor.


    See [Perez-Cruz et al., 2000]
    (http://ieeexplore.ieee.org/document/7075361/)
    ([pdf](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7075361)).
    """

    def __init__(self, C = 1, epsilon = 0.1, gamma = 1.0):
        self.C = C
        self.epsilon = epsilon
        self.kernel = RBF(gamma = gamma)

    def fit(self, x_train, y_train, max_iter = 100, method = 0):
        H = self.kernel(x_train, x_train)
        a = _np.zeros(len(x_train))
        a[0::2] = self.C
        a_star = _np.zeros(len(x_train))
        a_star[1::2] = self.C
        G13 = _np.zeros(len(y_train))
        Gb = _np.zeros(1)
        gamma = _np.zeros(len(x_train))

        S_ind = _np.array([1]*len(x_train))

        iter_counter = 0
        converged = False
        L = []

        while not converged:
            num_S1 = _np.sum(S_ind == 1)
            mat = _np.zeros((num_S1 + 1, num_S1 + 1))
            mat[:-1, :-1] = H[_np.logical_and.outer(S_ind == 1, S_ind == 1)].reshape((num_S1 ,num_S1)) + \
                _np.diag(1/(a + a_star)[S_ind == 1])
            mat[-1, :-1] = 1
            # Variation from paper: this column is set to +1 instead of -1
            mat[:-1, -1] = 1
            # Variation from paper: target vector contains +y_train instead of -y_train
            target = _np.concatenate([(a - a_star)[S_ind == 1]/(a + a_star)[S_ind == 1]*self.epsilon + y_train[S_ind == 1] - G13, Gb])

            gamma_b = _np.linalg.solve(mat, target)

            gamma[S_ind == 1] = gamma_b[:-1]
            gamma[S_ind == 2] = 0.0
            b = gamma_b[-1]

            # Variation from paper: flipped sign for epsilon
            e = H.dot(gamma) + b - y_train - self.epsilon
            e_star = y_train - H.dot(gamma) - b - self.epsilon

            # Calculate Lagrange at this point:
            L.append(0.5 * gamma.T.dot(H).dot(gamma) - (a.dot(e**2) + a_star.dot(e_star**2)))

            # Variation from paper: dropping factor 2
            a = _np.minimum(_np.maximum(0, self.C/e), 1e6)
            a_star = _np.minimum(_np.maximum(0, self.C/e_star), 1e6)

            # Reorder samples
            S_ind[_np.logical_and(S_ind == 3,
                _np.logical_and(e < 0.0, e_star < 0.0))] = 2
            S_ind[_np.logical_and(S_ind == 1,
                _np.abs(gamma) == self.C)] = 3
            S_ind[_np.logical_and(S_ind == 1,
                _np.logical_and(a == 0.0, a_star == 0.0))] = 2
            S_ind[_np.logical_and(S_ind == 2,
                _np.logical_or(a != 0.0, a_star != 0.0))] = 1

            G13 = H[_np.logical_and.outer(S_ind == 1, S_ind == 3)].reshape((num_S1, _np.sum(S_ind == 3))).dot(gamma[S_ind == 3])
            if any(G13 != 0.0):
                print G13
            Gb = -_np.sum(gamma[S_ind == 3], keepdims = True)

            if iter_counter > 0:
                if _np.linalg.norm(gamma - gamma_old) < 1e-10 and _np.abs(b - b_old) < 1e-10:
                    print("Converged after {} iterations".format(iter_counter))
                    converged = True

            if iter_counter >= max_iter:
                print("Maximum iterations ({}) reached!".format(max_iter))
                print("Final Deltas: gammma: {: 14.12f}, b: {: 14.12f}".format(
                    _np.linalg.norm(gamma - gamma_old), _np.abs(b - b_old)))
                converged = True

            gamma_old = gamma.copy()
            b_old = b.copy()
            iter_counter += 1

        self.x_train = x_train[S_ind == 1]
        self.gamma = gamma[S_ind == 1]
        self.intercept = b
        plt.semilogy(L)

    def predict(self, x_test):
        return self.kernel(x_test, self.x_train).dot(self.gamma) + self.intercept
