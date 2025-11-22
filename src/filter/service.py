import re
from typing import List, Pattern


class FilterService:
    def __init__(self, patterns: List[str]):
        self.patterns: List[Pattern] = [
            re.compile(p, flags=re.IGNORECASE) for p in patterns
        ]

    def filter(self, text: str) -> bool:
        """
        Returns True if ANY pattern matches the text.
        Streaming layer can use this to avoid sending the word back.
        """
        return any(p.search(text) for p in self.patterns)

    def apply(self, text: str, replacement: str = "zzzz") -> str:
        """
        Optional anonymizer (unchanged logic).
        """
        cleaned = text
        for p in self.patterns:
            cleaned = p.sub(replacement, cleaned)
        return cleaned
