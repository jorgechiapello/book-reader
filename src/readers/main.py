from abc import ABC, abstractmethod
from typing import List


class ReaderInterface(ABC):
    @abstractmethod
    def extract_pages(self) -> List[str]:
        pass
