from abc import ABC, abstractmethod
import numpy as np
from omegaconf import DictConfig


class BaseAPI(ABC):
    def __init__(self, config:DictConfig):
        self.config = config
    
    @abstractmethod
    def get_tldr(self, system_prompt, prompt) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def get_affiliations(self, system_prompt, prompt) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def get_embedding(self, system_prompt, prompt) -> np.ndarray:
        raise NotImplementedError