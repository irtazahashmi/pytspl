"""Module for serializing a kernel using the kernel type and data name."""

import gpytorch
import numpy as np

from pytspl.hogde_gp.kernels import Kernels


class KernelSerializer:

    def serialize(
        self, eigenpairs: np.ndarray, kernel_type: str, data_name: str
    ) -> gpytorch.kernels.Kernel:
        """
        Serialize the kernel using the kernel type and data name.

        Args:
            eigenpairs (np.ndarray):
            kernel_type (str): The kernel type to serialize.
            data_name (str): The data name for the kernel.

        Returns:
            gpytorch.kernels.Kernel: The serialized kernel.
        """
        kernel_serializer = self._get_serializer(
            kernel_type=kernel_type, data_name=data_name
        )
        return kernel_serializer(eigenpairs=eigenpairs)

    def _get_serializer(self, kernel_type: str, data_name: str):
        kernel_names = Kernels.get_names()

        # check if the kernel type is valid
        if kernel_type not in kernel_names:
            raise ValueError(
                f"Invalid kernel type. Choose from {kernel_names}"
            )

        if data_name == "forex":
            kernels = Kernels.get_forex_kernerls()
            kernel = kernels[kernel_type]
            return kernel
        else:
            raise ValueError("Invalid data name. Choose from ['forex']")
