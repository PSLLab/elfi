"""This module contains an interface for using the GPy library in ELFI."""

# TODO: make own general GPRegression and kernel classes

import copy
import logging

import GPy
import numpy as np
import time
from scipy import special

logger = logging.getLogger(__name__)
logging.getLogger("GP").setLevel(logging.WARNING)  # GPy library logger


#Dirty hack to make GPy work for us (we want N to be 1, not [[1]]) (bug in GPy!)
# from https://github.com/esiivola/vdsobo/blob/master/optimization.py
def logpdf_link(self, inv_link_f, y, Y_metadata=None):
    # I think this only alters the next line and if for when 'trials' is not present
    N = np.ones(y.shape) if Y_metadata is None else Y_metadata.get('trials', np.ones(y.shape))
    # logger.debug('logpdf N: {}'.format(N))
    np.testing.assert_array_equal(N.shape, y.shape)

    nchoosey = special.gammaln(N+1) - special.gammaln(y+1) - special.gammaln(N-y+1)
    Ny = N-y
    t1 = np.zeros(y.shape)
    t2 = np.zeros(y.shape)
    t1[y>0] = y[y>0]*np.log(inv_link_f[y>0])
    t2[Ny>0] = Ny[Ny>0]*np.log(1.-inv_link_f[Ny>0])
    
    return nchoosey + t1 + t2
GPy.likelihoods.Binomial.logpdf_link = logpdf_link #This is the dirty part

class GPyRegression:
    """Gaussian Process regression using the GPy library.

    GPy API: https://sheffieldml.github.io/GPy/
    """

    def __init__(self,
                 parameter_names=None,
                 bounds=None,
                 optimizer="scg",
                 max_opt_iters=50,
                 gp=None,
                 normalize=False,
                 **gp_params):
        """Initialize GPyRegression.

        Parameters
        ----------
        parameter_names : list of str, optional
            Names of parameter nodes. If None, sets dimension to 1.
        bounds : dict, optional
            The region where to estimate the posterior for each parameter in
            model.parameters.
            `{'parameter_name':(lower, upper), ... }`
            If not supplied, defaults to (0, 1) bounds for all dimensions.
        optimizer : string, optional
            Optimizer for the GP hyper parameters
            Alternatives: "scg", "fmin_tnc", "simplex", "lbfgsb", "lbfgs", "sgd"
            See also: paramz.Model.optimize()
        max_opt_iters : int, optional
        gp : GPy.model.GPRegression instance, optional
        **gp_params
            kernel : GPy.Kern
            noise_var : float
            mean_function

        """
        if parameter_names is None:
            input_dim = 1
        elif isinstance(parameter_names, (list, tuple)):
            input_dim = len(parameter_names)
        else:
            raise ValueError("Keyword `parameter_names` must be a list of strings")

        if bounds is None:
            logger.warning('Parameter bounds not specified. Using [0,1] for each parameter.')
            bounds = [(0, 1)] * input_dim
        elif len(bounds) != input_dim:
            raise ValueError(
                'Length of `bounds` ({}) does not match the length of `parameter_names` ({}).'
                .format(len(bounds), input_dim))

        elif isinstance(bounds, dict):
            if len(bounds) == 1:  # might be the case parameter_names=None
                bounds = [bounds[n] for n in bounds.keys()]
            else:
                # turn bounds dict into a list in the same order as parameter_names
                bounds = [bounds[n] for n in parameter_names]
        else:
            raise ValueError("Keyword `bounds` must be a dictionary "
                             "`{'parameter_name': (lower, upper), ... }`")

        self.parameter_names = parameter_names
        self.input_dim = input_dim
        self.bounds = bounds

        self.gp_params = gp_params

        self.optimizer = optimizer
        self.max_opt_iters = max_opt_iters

        self._gp = gp

        self._rbf_is_cached = False
        self.is_sampling = False  # set to True once in sampling phase
        self.virtual_deriv = False
        self.virtX = []
        self.virtY = []
        self.standardized_virtY = []
        self.normalize = normalize
        self.max_ep_iters = 1e4

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.virtual_deriv: # ep stuff doesn't pickle usably
            del state['_gp']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.virtual_deriv:
            # init gp
            kern = self.kernel.copy()
            kern_list = [kern] + [GPy.kern.DiffKern(kern,i) for i in range(self.input_dim)]
            lik_list = [GPy.likelihoods.Gaussian()]
            probit = GPy.likelihoods.Binomial(gp_link = GPy.likelihoods.link_functions.ScaledProbit(nu=1000))
            lik_list += [probit for i in range(self.input_dim)]
            if self.normalize:
                y_mean = self.virtY[0].mean(axis=0)
                y_std = self.virtY[0].std(axis=0)
                if np.any(y_std == 0):
                    logger.debug('Y has some zero sd {}'.format(y_std))
                    y_std[np.where(y_std==0)] = 1
                self.standardized_virtY = self.virtY.copy()
                self.standardized_virtY[0] = (self.virtY[0] - y_mean) / y_std 
                self._gp = GPy.models.MultioutputGP(X_list = self.virtX, Y_list = self.standardized_virtY,
                                                    kernel_list=kern_list, likelihood_list=lik_list,
                                                    inference_method=GPy.inference.latent_function_inference.EP(max_iters=self.max_ep_iters))
            else:
                self._gp = GPy.models.MultioutputGP(X_list = self.virtX, Y_list = self.virtY,
                                                    kernel_list=kern_list, likelihood_list=lik_list,
                                                    inference_method=GPy.inference.latent_function_inference.EP(max_iters=self.max_ep_iters))
            self.optimize()

    def __str__(self):
        """Return GPy's __str__."""
        return self._gp.__str__()

    def __repr__(self):
        """Return GPy's __str__."""
        return self.__str__()

    def predict(self, x, noiseless=False, reverse_normalize=False):
        """Return the GP model mean and variance at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]
        noiseless : bool
            whether to include the noise variance or not to the returned variance

        Returns
        -------
        tuple
            GP (mean, var) at x where
                mean : np.array
                    with shape (x.shape[0], 1)
                var : np.array
                    with shape (x.shape[0], 1)

        """
        # Ensure it's 2d for GPy
        x = np.asanyarray(x).reshape((-1, self.input_dim))

        if self._gp is None:
            # TODO: return from GP mean function if given
            return np.zeros((x.shape[0], 1)), \
                np.ones((x.shape[0], 1))

        # direct (=faster) implementation for RBF kernel
        if self.is_sampling and self._kernel_is_default:
            if not self._rbf_is_cached:
                self._cache_RBF_kernel()

            r2 = np.sum(x**2., 1)[:, None] + self._rbf_x2sum - 2. * x.dot(self._gp.X.T)
            kx = self._rbf_var * np.exp(r2 * self._rbf_factor) + self._rbf_bias
            mu = kx.dot(self._rbf_woodbury)

            var = self._rbf_var + self._rbf_bias
            var -= kx.dot(self._rbf_woodbury_inv.dot(kx.T))
            var += self._rbf_noisevar  # likelihood

            return mu, var
        else:
            self._rbf_is_cached = False  # in case one resumes fitting the GP after sampling

        if self.virtual_deriv:
            x = [x]
        if noiseless:
            # logger.debug('model predict output: {}'.format(self._gp.predict_noiseless(x)))
            mu, var = self._gp.predict_noiseless(x)
        else:
            # logger.debug('model predict output: {}'.format(self._gp.predict(x)))
            mu, var = self._gp.predict(x)
        if self.normalize and reverse_normalize:
            if self.virtual_deriv:
                # normal GP handles reverse already
                y_mean = self.virtY[0].mean(axis=0)
                y_std = self.virtY[0].std(axis=0)
            else:
                y_mean = self.virtY.mean(axis=0)
                y_std = self.virtY.std(axis=0)
            mu = mu*y_std + y_mean
            var = var*(y_std**2)
        return mu, var

    # TODO: find a more general solution
    # cache some RBF-kernel-specific values for faster sampling
    def _cache_RBF_kernel(self):
        self._rbf_var = float(self._gp.kern.rbf.variance)
        self._rbf_factor = -0.5 / float(self._gp.kern.rbf.lengthscale)**2
        self._rbf_bias = float(self._gp.kern.bias.K(self._gp.X)[0, 0])
        self._rbf_noisevar = float(self._gp.likelihood.variance[0])
        self._rbf_woodbury = self._gp.posterior.woodbury_vector
        self._rbf_woodbury_inv = self._gp.posterior.woodbury_inv
        self._rbf_woodbury_chol = self._gp.posterior.woodbury_chol
        self._rbf_x2sum = np.sum(self._gp.X**2., 1)[None, :]
        self._rbf_is_cached = True

    def predict_mean(self, x):
        """Return the GP model mean function at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]

        Returns
        -------
        np.array
            with shape (x.shape[0], 1)

        """
        return self.predict(x)[0]

    def predictive_gradients(self, x):
        """Return the gradients of the GP model mean and variance at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]

        Returns
        -------
        tuple
            GP (grad_mean, grad_var) at x where
                grad_mean : np.array
                    with shape (x.shape[0], input_dim)
                grad_var : np.array
                    with shape (x.shape[0], input_dim)

        """
        # Ensure it's 2d for GPy
        x = np.asanyarray(x).reshape((-1, self.input_dim))

        if self._gp is None:
            # TODO: return from GP mean function if given
            return np.zeros((x.shape[0], self.input_dim)), \
                np.zeros((x.shape[0], self.input_dim))

        # direct (=faster) implementation for RBF kernel
        if self.is_sampling and self._kernel_is_default:
            if not self._rbf_is_cached:
                self._cache_RBF_kernel()

            r2 = np.sum(x**2., 1)[:, None] + self._rbf_x2sum - 2. * x.dot(self._gp.X.T)
            kx = self._rbf_var * np.exp(r2 * self._rbf_factor)
            dkdx = 2. * self._rbf_factor * (x - self._gp.X) * kx.T
            grad_mu = dkdx.T.dot(self._rbf_woodbury).T

            v = np.linalg.solve(self._rbf_woodbury_chol, kx.T + self._rbf_bias)
            dvdx = np.linalg.solve(self._rbf_woodbury_chol, dkdx)
            grad_var = -2. * dvdx.T.dot(v).T
        else:
            if self.virtual_deriv:
                grad_mu, grad_var = self._gp.predictive_gradients([x])
            else:
                grad_mu, grad_var = self._gp.predictive_gradients(x)
                grad_mu = grad_mu[:, :, 0]  # Assume 1D output (distance in ABC)
            # logger.debug('predictive gradient output: {}\n{}'.format(grad_mu, grad_var))

        return grad_mu, grad_var

    def predictive_gradient_mean(self, x):
        """Return the gradient of the GP model mean at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]

        Returns
        -------
        np.array
            with shape (x.shape[0], input_dim)

        """
        return self.predictive_gradients(x)[0]

    def _init_gp(self, x, y):
        self._kernel_is_default = False

        if self.gp_params.get('kernel') is None:
            kernel = self._default_kernel(x, y)

            if self.gp_params.get('noise_var') is None and self.gp_params.get(
                    'mean_function') is None:
                self._kernel_is_default = True
        else:
            kernel = self.gp_params.get('kernel')

        self.kernel = kernel

        if self.virtual_deriv:
            self.virtX = [x] + [np.empty((0,self.input_dim))]*self.input_dim
            self.virtY = [y] + [np.empty((0,1))]*self.input_dim
            kern = self.kernel.copy()
            kern_list = [kern] + [GPy.kern.DiffKern(kern,i) for i in range(self.input_dim)]
            lik_list = [GPy.likelihoods.Gaussian()]
            probit = GPy.likelihoods.Binomial(gp_link = GPy.likelihoods.link_functions.ScaledProbit(nu=1000))
            lik_list += [probit for i in range(self.input_dim)]
            start = time.time()
            # need to be able to add in or remove virtual observations
            if self.normalize:
                y_mean = self.virtY[0].mean(axis=0)
                y_std = self.virtY[0].std(axis=0)
                if np.any(y_std == 0):
                    logger.debug('Y has some zero sd {}'.format(y_std))
                    y_std[np.where(y_std==0)] = 1
                self.standardized_virtY = self.virtY.copy()
                self.standardized_virtY[0] = (self.virtY[0] - y_mean) / y_std
                self._gp = GPy.models.MultioutputGP(X_list = self.virtX, Y_list = self.standardized_virtY,
                                                    kernel_list=kern_list, likelihood_list=lik_list,
                                                    inference_method=GPy.inference.latent_function_inference.EP(max_iters=self.max_ep_iters))
            else:
                self._gp = GPy.models.MultioutputGP(X_list = self.virtX, Y_list = self.virtY,
                                                    kernel_list=kern_list, likelihood_list=lik_list,
                                                    inference_method=GPy.inference.latent_function_inference.EP(max_iters=self.max_ep_iters))
            end = time.time()
            logger.debug("Creating GP took: {}".format(str(end-start)))
            start = time.time()
            self.optimize()
            end = time.time()
            logger.debug("Optimizing GP took: {}".format(str(end-start)))
        else:
            self.virtX = x
            self.virtY = y
            kern = self.kernel.copy()
            noise_var = self.gp_params.get('noise_var') or np.max(y)**2. / 100.
            mean_function = self.gp_params.get('mean_function')
            noise_prior = self.gp_params.get('noise_prior')
            if self.normalize:
                y_mean = self.virtY.mean(axis=0)
                y_std = self.virtY.std(axis=0)
                if np.any(y_std == 0):
                    logger.debug('Y has some zero sd {}'.format(y_std))
                    y_std[np.where(y_std==0)] = 1
                self.standardized_virtY = self.virtY.copy()
                self.standardized_virtY = (self.virtY - y_mean) / y_std
                self._gp = self._make_gpy_instance(
                    self.virtX, self.standardized_virtY, kernel=kern, noise_var=noise_var, mean_function=mean_function)
            else:
                self._gp = self._make_gpy_instance(
                    self.virtX, self.virtY, kernel=kern, noise_var=noise_var, mean_function=mean_function)
            if noise_prior is not None:
                self._gp.Gaussian_noise.set_prior(noise_prior)
            self.optimize()

    def _default_kernel(self, x, y):
        # Some heuristics to choose kernel parameters based on the initial data
        length_scale = (np.max(self.bounds) - np.min(self.bounds)) / 3.
        kernel_var = (np.max(y) / 3.)**2.
        bias_var = kernel_var / 4.

        # Construct a default kernel
        kernel = GPy.kern.RBF(input_dim=self.input_dim)

        # Set the priors
        kernel.lengthscale.set_prior(
            GPy.priors.Gamma.from_EV(length_scale, length_scale), warning=False)
        kernel.variance.set_prior(GPy.priors.Gamma.from_EV(kernel_var, kernel_var), warning=False)

        # If no mean function is specified, add a bias term to the kernel
        if 'mean_function' not in self.gp_params:
            bias = GPy.kern.Bias(input_dim=self.input_dim)
            bias.set_prior(GPy.priors.Gamma.from_EV(bias_var, bias_var), warning=False)
            kernel += bias

        return kernel

    def _make_gpy_instance(self, x, y, kernel, noise_var, mean_function):
        # if self.normalize:
        #     return GPy.models.GPRegression(
        #         X=x, Y=y, kernel=kernel, noise_var=noise_var, mean_function=mean_function,
        #         normalizer=True)
        # else:
        return GPy.models.GPRegression(
            X=x, Y=y, kernel=kernel, noise_var=noise_var, mean_function=mean_function)

    # def get_model_likelihood(self, noise = 0.0):
    #     '''
    #     Returns gaussian likelihood with fixed noise.
    #     '''
    #     lik = GPy.likelihoods.Gaussian(variance=noise)
    #     if noise < 0.00001:
    #         lik.variance.constrain_fixed(value=1e-6,warning=True,trigger_parent=True)
    #     else:
    #         lik.variance.constrain_fixed(value=noise,warning=True,trigger_parent=True)
    #     return lik

    def update(self, x, y, update_gp=False):
        """Update the GP model with new data.

        Parameters
        ----------
        x : np.array
        y : np.array
        optimize : bool, optional
            Whether to optimize hyperparameters.

        """
        # Must cast these as 2d for GPy
        x = x.reshape((-1, self.input_dim))
        y = y.reshape((-1, 1))

        if self._gp is None:
            self._init_gp(x, y)
        elif self.virtual_deriv:
            self.virtX[0] = np.r_[self.virtX[0], x]
            self.virtY[0] = np.r_[self.virtY[0], y]
            if update_gp:
                # not certain about the model and variance here compared to the bolfi code
                kern = self.kernel.copy()
                kern_list = [kern] + [GPy.kern.DiffKern(kern,i) for i in range(self.input_dim)]
                lik_list = [GPy.likelihoods.Gaussian()]
                probit = GPy.likelihoods.Binomial(gp_link = GPy.likelihoods.link_functions.ScaledProbit(nu=1000))
                lik_list += [probit for i in range(self.input_dim)]
                start = time.time()
                if self.normalize:
                    y_mean = self.virtY[0].mean(axis=0)
                    y_std = self.virtY[0].std(axis=0)
                    if np.any(y_std == 0):
                        logger.debug('Y has some zero sd {}'.format(y_std))
                        y_std[np.where(y_std==0)] = 1
                    self.standardized_virtY = self.virtY.copy()
                    self.standardized_virtY[0] = (self.virtY[0] - y_mean) / y_std
                    self._gp = GPy.models.MultioutputGP(X_list = self.virtX, Y_list = self.standardized_virtY,
                                                        kernel_list=kern_list, likelihood_list=lik_list,
                                                        inference_method=GPy.inference.latent_function_inference.EP(max_iters=self.max_ep_iters))
                else:
                    self._gp = GPy.models.MultioutputGP(X_list = self.virtX, Y_list = self.virtY,
                                                        kernel_list=kern_list, likelihood_list=lik_list,
                                                        inference_method=GPy.inference.latent_function_inference.EP(max_iters=self.max_ep_iters))
                end = time.time()
                logger.debug("Creating GP took: {}".format(str(end-start)))
                start = time.time()
                self.optimize()
                end = time.time()
                logger.debug("Optimizing GP took: {}".format(str(end-start)))
        else:
            # Reconstruct with new data
            self.virtX = np.r_[self.virtX, x]
            self.virtY = np.r_[self.virtY, y]
            if update_gp:
                # It seems that GPy will do some optimization unless you make copies of everything
                kernel = self._gp.kern.copy() if self._gp.kern else None
                noise_var = self._gp.Gaussian_noise.variance[0]
                mean_function = self._gp.mean_function.copy() if self._gp.mean_function else None
                if self.normalize:
                        y_mean = self.virtY.mean(axis=0)
                        y_std = self.virtY.std(axis=0)
                        if np.any(y_std == 0):
                            logger.debug('Y has some zero sd {}'.format(y_std))
                            y_std[np.where(y_std==0)] = 1
                        self.standardized_virtY = self.virtY.copy()
                        self.standardized_virtY = (self.virtY - y_mean) / y_std
                        self._gp = self._make_gpy_instance(
                            self.virtX, self.standardized_virtY, kernel=kernel, noise_var=noise_var, mean_function=mean_function)
                else:
                    self._gp = self._make_gpy_instance(
                            self.virtX, self.virtY, kernel=kernel, noise_var=noise_var, mean_function=mean_function)
                self.optimize()

    def update_virt(self):
        if not self.virtual_deriv:
            raise Exception('update_virt is not valid for a model without virtual observations')

        if self._gp is None:
            raise Exception('cannot do virtual update without an existing GP')
        else:
            kern = self.kernel.copy()
            kern_list = [kern] + [GPy.kern.DiffKern(kern,i) for i in range(self.input_dim)]
            lik_list = [GPy.likelihoods.Gaussian()]
            probit = GPy.likelihoods.Binomial(gp_link = GPy.likelihoods.link_functions.ScaledProbit(nu=1000))
            lik_list += [probit for i in range(self.input_dim)]
            start = time.time()
            if self.normalize:
                y_mean = self.virtY[0].mean(axis=0)
                y_std = self.virtY[0].std(axis=0)
                if np.any(y_std == 0):
                    logger.debug('Y has some zero sd {}'.format(y_std))
                    y_std[np.where(y_std==0)] = 1
                self.standardized_virtY = self.virtY.copy()
                self.standardized_virtY[0] = (self.virtY[0] - y_mean) / y_std
                self._gp = GPy.models.MultioutputGP(X_list = self.virtX, Y_list = self.standardized_virtY,
                                                    kernel_list=kern_list, likelihood_list=lik_list,
                                                    inference_method=GPy.inference.latent_function_inference.EP(max_iters=self.max_ep_iters))
            else:
                self._gp = GPy.models.MultioutputGP(X_list = self.virtX, Y_list = self.virtY,
                                                    kernel_list=kern_list, likelihood_list=lik_list,
                                                    inference_method=GPy.inference.latent_function_inference.EP(max_iters=self.max_ep_iters))
            end = time.time()
            logger.debug("Creating GP took: {}".format(str(end-start)))
            start = time.time()
            self.optimize()
            end = time.time()
            logger.debug("Optimizing GP took: {}".format(str(end-start)))
    
    def extract_simple_model(self):
        kern = self.kernel.copy()
        self._gp = GPy.models.GPRegression(X=self.virtX[0], Y=self.virtY[0], kernel=kern, normalizer=True)
        self.optimize()
        self.virtual_deriv = False
                        
    def optimize(self):
        """Optimize GP hyperparameters."""
        logger.debug("Optimizing GP hyperparameters")
        try:
            self._gp.optimize(self.optimizer, max_iters=self.max_opt_iters)
        except np.linalg.linalg.LinAlgError:
            logger.warning("Numerical error in GP optimization. Stopping optimization")

    def mean_std(self):
        if not self.normalize:
            # what does standard gp with normalizer return?
            logger.warning('mean_std only makes sense to use if Y is being standardized')
            return None, None
        if self.virtual_deriv:
            return self.virtY[0].mean(axis=0), self.virtY[0].std(axis=0)
        else:
            return self.virtY.mean(axis=0), self.virtY.std(axis=0)

    @property
    def n_evidence(self):
        """Return the number of observed samples."""
        if self._gp is None:
            return 0
        return self._gp.num_data

    @property
    def X(self):
        """Return input evidence."""
        if self.virtual_deriv:
            # logger.debug('GP X output: {}'.format(self.virtX[0]))
            return self.virtX[0]
        else:
            return self._gp.X

    @property
    def Y(self):
        """Return output evidence."""
        if self.virtual_deriv:
            return self.virtY[0]
        else:
            return self._gp.Y

    @property
    def noise(self):
        """Return the noise."""
        return self._gp.Gaussian_noise.variance[0]

    @property
    def instance(self):
        """Return the gp instance."""
        return self._gp

    def copy(self):
        """Return a copy of current instance."""
        kopy = copy.copy(self)
        if self._gp:
            kopy._gp = self._gp.copy()

        if 'kernel' in self.gp_params:
            kopy.gp_params['kernel'] = self.gp_params['kernel'].copy()

        if 'mean_function' in self.gp_params:
            kopy.gp_params['mean_function'] = self.gp_params['mean_function'].copy()

        return kopy

    def __copy__(self):
        """Return a copy of current instance."""
        return self.copy()
