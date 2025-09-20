from typing import List
from src.core.drift import get_log_probs

class BaseVector:
    def __init__(self, vector):
        self.vector = vector

    def evaluate(self, text: str) -> float:
        return await get_log_probs(self.vector, text)        
    
    def train(self, data: List[str]) -> None:
        pass
    
    def get_vector(self) -> List[float]:
        pass