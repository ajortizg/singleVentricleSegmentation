import torch
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Iterable, Any, Dict, Optional, Tuple

from svs.utils.constants import *
from svs.modules.transforms.functional import (
    add_dim_at,
    reorder_axes,
    remove_dim_at,
    ensure_float
)


class BaseTransform(object, metaclass=ABCMeta):
    """
    Base class for data transformation.

    This class provides an interface for implementing data transformations
    that can be applied to specified keys within a data dictionary.
    """

    @abstractmethod
    def __init__(self, keys: Iterable[str]):
        """
        Initializes the BaseTransform.

        Args:
            keys (Iterable[str]): List of keys in the data dictionary to which the transformation will be applied.
        """
        self.keys = keys

    def apply_transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.keys:
            x = data[key]

            # Get the metada if available
            metadata_key = key + METADATA_SUBFIX
            metadata = data[metadata_key] if metadata_key in data else None

            # Apply transformation
            x_new = self._transform_impl(x, key, metadata)
            data[key] = x_new
        return data

    @ abstractmethod
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @ abstractmethod
    def _transform_impl(self, x: Any, key: str, metadata: Optional[Dict[str, Any]] = None):
        pass

    def class_name(self):
        return str(type(self).__name__)

    def items(self):
        return self.__dict__

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
        return ret_str


class Compose:
    def __init__(self, transforms: Iterable[BaseTransform]):
        self.transforms = transforms

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class OneOf:
    def __init__(self, transforms: Iterable[BaseTransform]):
        self.transforms = transforms
        self.n = len(self.transforms)

    def __call__(self, data):
        idx = np.random.randint(self.n)
        return self.transforms[idx](data)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class DoNothing(BaseTransform):
    def __init__(self):
        super().__init__(None)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data

    def _transform_impl(self, x: Any, key: str, metadata: Optional[Dict[str, Any]] = None):
        return x


class ToTensor(BaseTransform):
    def __init__(self, keys: Iterable[str], dtype=None):
        super(ToTensor, self).__init__(keys)
        self.dtype = dtype

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return super().apply_transform(data)

    def _transform_impl(self, x: Any, key: str, metadata: Optional[Dict[str, Any]] = None):
        if isinstance(x, np.ndarray):
            y = torch.from_numpy(x)
        elif (isinstance(x, (list, tuple))):
            y = torch.tensor(x)
        else:
            raise TypeError(f"type of x ({type(x)}) is not supported.")
        return y.to(self.dtype) if self.dtype is not None else y


class AddDimAt(BaseTransform):
    def __init__(self, keys: Iterable[str], axis: int):
        super().__init__(keys)
        self.axis = axis

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return super().apply_transform(data)

    def _transform_impl(self, x: Any, key: str, metadata: Optional[Dict[str, Any]] = None):
        return add_dim_at(self.axis, x)


class RemoveDimAt(BaseTransform):
    def __init__(self, keys: Iterable[str], axis: int):
        super().__init__(keys)
        self.axis = axis

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return super().apply_transform(data)

    def _transform_impl(self, x: Any, key: str, metadata: Optional[Dict[str, Any]] = None):
        return remove_dim_at(self.axis, x)


class ReorderAxes(BaseTransform):
    def __init__(self, keys: Iterable[str], axes: Tuple[int]):
        super().__init__(keys)
        self.axes = axes

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return super().apply_transform(data)

    def _transform_impl(self, x: Any, key: str, metadata: Optional[Dict[str, Any]] = None):
        return reorder_axes(self.axes, x)


class EnsureFloat(BaseTransform):
    def __init__(self, keys: Iterable[str]):
        super().__init__(keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return super().apply_transform(data)

    def _transform_impl(self, x: Any, key: str, metadata: Optional[Dict[str, Any]] = None):
        return ensure_float(x)
