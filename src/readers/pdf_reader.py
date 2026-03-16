from typing import List

from readers.main import ReaderInterface


class PyPdfReader(ReaderInterface):
    def __init__(self, path: str):
        from pypdf import PdfReader
        self.reader = PdfReader(path)

    def extract_pages(self) -> List[str]:
        return [page.extract_text() or "" for page in self.reader.pages]
