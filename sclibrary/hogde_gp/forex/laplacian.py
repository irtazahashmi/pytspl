from typing import Optional

import torch
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel


class LaplacianKernelForex(Kernel):
    def __init__(
        self, eigenpairs: list[torch.tensor], kappa_bounds=(1e-5, 1e5)
    ):
        super().__init__()
        (
            self.harm_eigvects,
            self.grad_eigvects,
            self.curl_eigvects,
            self.harm_eigvals,
            self.grad_eigvals,
            self.curl_eigvals,
        ) = eigenpairs

        # register the raw parameters
        self.register_parameter(
            name="raw_kappa_down",
            parameter=torch.nn.Parameter(torch.zeros(1, 1)),
        )
        self.register_parameter(
            name="raw_kappa_up",
            parameter=torch.nn.Parameter(torch.zeros(1, 1)),
        )
        self.register_parameter(
            name="raw_mu", parameter=torch.nn.Parameter(torch.zeros(1, 1))
        )
        # set the kappa constraints
        self.register_constraint("raw_kappa_down", Positive())
        self.register_constraint("raw_kappa_up", Positive())
        self.register_constraint("raw_mu", Positive())

    # set up the actual parameters
    @property
    def kappa_down(self):
        return self.raw_kappa_down_constraint.transform(self.raw_kappa_down)

    @kappa_down.setter
    def kappa_down(self, value):
        self._set_kappa_down(value)

    def _set_kappa_down(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa_down)
        self.initialize(
            raw_kappa_down=self.raw_kappa_down_constraint.inverse_transform(
                value
            )
        )

    # set up the actual parameters
    @property
    def mu(self):
        return self.raw_mu_constraint.transform(self.raw_mu)

    @mu.setter
    def mu(self, value):
        self._set_mu(value)

    def _set_mu(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mu)
        self.initialize(raw_mu=self.raw_mu_constraint.inverse_transform(value))

    @property
    def kappa_up(self):
        return self.raw_kappa_up_constraint.transform(self.raw_kappa_up)

    @kappa_up.setter
    def kappa_up(self, value):
        self._set_kappa_up(value)

    def _set_kappa_up(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa_up)
        self.initialize(
            raw_kappa_up=self.raw_kappa_up_constraint.inverse_transform(value)
        )

    def _eval_covar_matrix(self):
        """Define the full covariance matrix -- full kernel matrix as
        a property to avoid repeative computation of the kernel matrix"""
        k0 = 1 / (1e-3 + self.harm_eigvals).squeeze()
        k1 = 1 / (self.grad_eigvals).squeeze()
        k2 = 1 / (self.curl_eigvals).squeeze()
        return (k0, k1, k2)

    @property
    def covar_matrix(self):
        return self._eval_covar_matrix()

    def forward(self, x1, x2=None, diag: Optional[bool] = False, **params):
        x1, x2 = x1.long(), x2.long()
        x1 = x1.squeeze(-1)
        x2 = x2.squeeze(-1)
        # compute the kernel matrix
        if x2 is None:
            x2 = x1

        (k0, k1, k2) = self._eval_covar_matrix()
        K0 = self.harm_eigvects[x1, :] * k0 @ self.harm_eigvects[x2, :].T
        K1 = self.grad_eigvects[x1, :] * k1 @ self.grad_eigvects[x2, :].T
        K2 = self.curl_eigvects[x1, :] * k2 @ self.curl_eigvects[x2, :].T
        K = self.mu * K0 + self.kappa_down * K1 + self.kappa_up * K2
        if diag:
            return K.diag()
        else:
            return K
