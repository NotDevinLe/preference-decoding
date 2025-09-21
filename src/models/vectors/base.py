from typing import List
import torch

class BaseVector:
    def __init__(self, vector=None):
        self.vector = vector

    def evaluate(self, text: str) -> float:
        """
        Evaluate text using the vector. 
        Base implementation - should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def train(self, data: List[str]) -> None:
        """
        Train the vector on data.
        Base implementation - should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement train method")
    
    def get_vector(self) -> torch.Tensor:
        """
        Get the current vector.
        """
        return self.vector
    
    def set_vector(self, vector: torch.Tensor) -> None:
        """
        Set the vector.
        """
        self.vector = vector