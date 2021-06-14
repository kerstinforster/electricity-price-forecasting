""" This file contains the base class to the data getters."""

from abc import ABC, abstractmethod
from pathlib import Path
import os


class DataGetter(ABC):
    """
    The abstract interface for all data getters
    """
    def __init__(self, name: str):
        """
        Constructor for the data getter setting the name field
        :param name: Name of the data getter
        """
        self.name = name
        self._root_dir = self.get_project_root()
        self.data_dir = os.path.join(self._root_dir, "data", name)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    @abstractmethod
    def get_data(self) -> None:
        """
        This function is implemented in derived classes to download the data
        :return: bool True if successful, otherwise False
        """
        raise NotImplementedError("This is an abstract class")

    @staticmethod
    def get_project_root() -> Path:
        """
        This function returns the project's root directory
        :return: The root path of the project
        """
        return Path(__file__).parent.parent.parent
