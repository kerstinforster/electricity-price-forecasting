""" This file contains the base class to the data getters."""

from abc import ABC, abstractmethod


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

    @abstractmethod
    def get_data(self) -> bool:
        """
        This function is implemented in derived classes to download the data
        :return: bool True if successful, otherwise False
        """
        raise NotImplementedError("This is an abstract class")
