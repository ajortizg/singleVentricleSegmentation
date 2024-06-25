import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Iterable, Any, Dict, Optional

from svs.utils.constants import *


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
        self.cur_key = None

    def apply_transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.keys:
            # Set current key and data
            self.cur_key = key
            x = data[key]

            # Get the metada if available
            metadata_key = key + METADATA_SUBFIX
            metadata = data[metadata_key] if metadata_key in data else None

            # Apply transformation
            x_new = self._transform_impl(x, metadata)
            data[key] = x_new
        return data

    @abstractmethod
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _transform_impl(self, x: Any, metadata: Optional[Dict[str, Any]] = None):
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

    def _transform_impl(self, x: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return x
