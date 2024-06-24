import numpy as np
import pytest
from gpytorch.kernels import Kernel

from sclibrary.hogde_gp.kernel_serializer import KernelSerializer


class TestKernelSerializer:

    def test_serialize(self):
        kernel_type = "diffusion"
        data_name = "forex"

        eigenpairs = np.asarray([[] for i in range(6)])

        kernel_serializer = KernelSerializer()
        kernel = kernel_serializer.serialize(
            eigenpairs=eigenpairs,
            kernel_type=kernel_type,
            data_name=data_name,
        )

        assert issubclass(kernel.__class__, Kernel)

    def test_invalid_kernel_type(self):
        kernel_type = "invalid"
        data_name = "forex"

        kernel_serializer = KernelSerializer()
        with pytest.raises(ValueError):
            kernel_serializer.serialize(
                eigenpairs=np.array([]),
                kernel_type=kernel_type,
                data_name=data_name,
            )

    def test_invalid_data_name(self):
        kernel_type = "diffusion"
        data_name = "invalid"

        kernel_serializer = KernelSerializer()
        with pytest.raises(ValueError):
            kernel_serializer.serialize(
                eigenpairs=np.array([]),
                kernel_type=kernel_type,
                data_name=data_name,
            )

    def test_get_serializer(self):
        kernel_type = "diffusion"
        data_name = "forex"

        kernel_serializer = KernelSerializer()
        serializer = kernel_serializer._get_serializer(
            kernel_type=kernel_type,
            data_name=data_name,
        )

        assert issubclass(serializer, Kernel)
