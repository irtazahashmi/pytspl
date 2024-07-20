"""Types of kernels for the Hodge Gaussian Process model."""

from enum import Enum

from .forex import (
    DiffusionKernelForex,
    DiffusionNonHCKernelForex,
    LaplacianKernelForex,
    MaternKernelForex,
    MaternNonHCKernelForex,
)


class Kernels(Enum):
    """
    Enum class for the implemented kernels.

    - DIFFUSION: Diffusion kernel.
    - DIFFUSION_NON_HC: Diffusion kernel without hyperparameters.
    - MATERN: Matern kernel.
    - MATERN_NON_HC: Matern kernel without hyperparameters.
    - LAPLACIAN: Laplacian kernel.
    - LAPLACIAN_NON_HC: Laplacian kernel without hyperparameters.
    """

    DIFFUSION = "diffusion"
    DIFFUSION_NON_HC = "diffusion_non_hc"

    MATERN = "matern"
    MATERN_NON_HC = "matern_non_hc"

    LAPLACIAN = "laplacian"
    LAPLACIAN_NON_HC = "laplacian_non_hc"

    @classmethod
    def get_names(cls):
        """Return the names of the implemented kernels."""
        return list(map(lambda c: c.value, cls))

    def get_forex_kernerls():
        """Return the implemented kernels for the forex data."""
        return {
            Kernels.DIFFUSION.value: DiffusionKernelForex,
            Kernels.DIFFUSION_NON_HC.value: DiffusionNonHCKernelForex,
            Kernels.MATERN.value: MaternKernelForex,
            Kernels.MATERN_NON_HC.value: MaternNonHCKernelForex,
            Kernels.LAPLACIAN.value: LaplacianKernelForex,
        }
