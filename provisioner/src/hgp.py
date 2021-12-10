import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, ConstantKernel as C
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
import Kernel, _approx_fprime, Hyperparameter, RBF
import pickle
from time import time
import random

def save_model(filename, gp_heteroscedastic):
    file_path = "checkpoints/" + filename + ".pt"
    with open(file_path, 'wb') as f:
        pickle.dump(gp_heteroscedastic, f)

def load_model(filename):
    dtl = filename.split('_')
    al = 1 if '10' in dtl[-1] else 0
    dataset, dataset_size, max_container_ips = eval("load_"+'_'.join(dtl[:-1])+"_data("+dtl[-1]+")")
    X = np.array([np.array(i[0]).reshape(-1) for i in dataset])
    y = np.array([Coeff_Energy*i[1][0] + Coeff_Latency*i[1][1] for i in dataset])
    prototypes = KMeans(n_clusters=10).fit(X).cluster_centers_
    kernel_hetero = C(1.0, (1e-10, 1000)) * RBF(1, (0.01, 100.0)) \
        + HeteroscedasticKernel.construct(prototypes, 1e-3, (1e-10, 50.0),
                                          gamma=5.0, gamma_bounds="fixed")
    gp_heteroscedastic = GaussianProcessRegressor(kernel=kernel_hetero, alpha=al)
    file_path1 = "checkpoints/" + filename + ".pt"
    file_path2 = 'scheduler/HGP/' + file_path1
    file_path = file_path1 if os.path.exists(file_path1) else file_path2
    if os.path.exists(file_path):
        print(color.GREEN+"Loading pre-trained model: "+filename+color.ENDC)
        with open(file_path, 'rb') as f:
            gp_heteroscedastic = pickle.load(f)
    else:
        print(color.GREEN+"Creating new model: "+filename+color.ENDC)
    return gp_heteroscedastic, X, y, max_container_ips

class HeteroscedasticKernel(Kernel):
    """Kernel which learns a heteroscedastic noise model.

    This kernel learns for a set of prototypes values from the data space
    explicit noise levels. These exemplary noise levels are then generalized to
    the entire data space by means for kernel regression.

    Parameters
    ----------
    prototypes : array-like, shape = (n_prototypes, n_X_dims)
        Prototypic samples from the data space for which noise levels are
        estimated.

    sigma_2 : float, default: 1.0
        Parameter controlling the initial noise level

    sigma_2_bounds : pair of floats >= 0, default: (0.1, 10.0)
        The lower and upper bound on sigma_2

    gamma : float, default: 1.0
        Length scale of the kernel regression on the noise level

    gamma_bounds : pair of floats >= 0, default: (1e-2, 1e2)
        The lower and upper bound on gamma
    """
    def __init__(self, prototypes, sigma_2=1.0, sigma_2_bounds=(0.1, 10.0),
                 gamma=1.0, gamma_bounds=(1e-2, 1e2)):
        assert prototypes.shape[0] == sigma_2.shape[0]
        self.prototypes = prototypes

        self.sigma_2 = np.asarray(sigma_2)
        self.sigma_2_bounds = sigma_2_bounds

        self.gamma = gamma
        self.gamma_bounds = gamma_bounds

        self.hyperparameter_sigma_2 = \
                Hyperparameter("sigma_2", "numeric", self.sigma_2_bounds,
                               self.sigma_2.shape[0])

        self.hyperparameter_gamma = \
                Hyperparameter("gamma", "numeric", self.gamma_bounds)

    @classmethod
    def construct(cls, prototypes, sigma_2=1.0, sigma_2_bounds=(0.1, 10.0),
                  gamma=1.0, gamma_bounds=(1e-2, 1e2)):
        prototypes = np.asarray(prototypes)
        if prototypes.shape[0] > 1 and len(np.atleast_1d(sigma_2)) == 1:
            sigma_2 = np.repeat(sigma_2, prototypes.shape[0])
            sigma_2_bounds = np.vstack([sigma_2_bounds] *prototypes.shape[0])
        return cls(prototypes, sigma_2, sigma_2_bounds, gamma, gamma_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        prototypes_std = self.prototypes.std(0)
        n_prototypes = self.prototypes.shape[0]
        n_gradient_dim = \
            n_prototypes + (0 if self.hyperparameter_gamma.fixed else 1)

        X = np.atleast_2d(X)
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K= np.eye(X.shape[0]) * self.diag(X)
            if eval_gradient:
                K_gradient = \
                    np.zeros((K.shape[0], K.shape[0], n_gradient_dim))
                K_pairwise = \
                    pairwise_kernels(self.prototypes / (prototypes_std + EPSILON),
                                     X / (prototypes_std + EPSILON),
                                     metric="rbf", gamma=self.gamma)
                for i in range(n_prototypes):
                    for j in range(K.shape[0]):
                        K_gradient[j, j, i] = \
                            self.sigma_2[i] * K_pairwise[i, j] \
                            / (K_pairwise[:, j].sum() + EPSILON)
                if not self.hyperparameter_gamma.fixed:
                    # XXX: Analytic expression for gradient?
                    def f(gamma):  # helper function
                        theta = self.theta.copy()
                        theta[-1] = gamma[0]
                        return self.clone_with_theta(theta)(X, Y)
                    K_gradient[:, :, -1] = \
                        _approx_fprime([self.theta[-1]], f, 1e-5)[:, :, 0]
                return K, K_gradient
            else:
                return K
        else:
            K = np.zeros((X.shape[0], Y.shape[0]))
            return K   # XXX: similar entries?

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return False

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        prototypes_std = self.prototypes.std(0)
        n_prototypes = self.prototypes.shape[0]

        # kernel regression of noise levels
        K_pairwise = \
            pairwise_kernels(self.prototypes / (prototypes_std + EPSILON),
                             X / (prototypes_std + EPSILON),
                             metric="rbf", gamma=self.gamma)

        return (K_pairwise * self.sigma_2[:, None]).sum(axis=0) \
                / (K_pairwise.sum(axis=0) + EPSILON)

    def __repr__(self):
        return "{0}(sigma_2=[{1}], gamma={2})".format(self.__class__.__name__,
            ", ".join(map("{0:.3g}".format, self.sigma_2)), self.gamma)

if __name__ == '__main__':
    data_type = argv[1] # can be 'energy_latency' + '_' + str(HOSTS)
    exec_type = argv[2] # can be 'train', 'opt'

    gp_heteroscedastic, X, y, max_container_ips = load_model(data_type)

    if exec_type == "train":
        gp_heteroscedastic.fit(X, y)
        print("Heteroscedastic kernel: %s" % gp_heteroscedastic.kernel_)
        print("Heteroscedastic LML: %.3f" \
            % gp_heteroscedastic.log_marginal_likelihood(gp_heteroscedastic.kernel_.theta))
        save_model(data_type, gp_heteroscedastic)

    else:
        init = random.choice(X)
        x, s = gp_heteroscedastic.predict(init.reshape(1, -1), return_std=True)
        print((x + UCB_K * s)[0])
        start = time()
        result, fitness = HGPopt(init, gp_heteroscedastic, data_type)
        print("Time", time()-start)
        print("Iteration: {}\nResult: {}\nFitness: {}".format(0, result, fitness)) 