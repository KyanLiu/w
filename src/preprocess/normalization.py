import re
import unicodedata

_WHITESPACE_PATTERN = re.compile(r"\s+")

def _replaceCharacter(ch: str) -> str:
    """ Keep whitespace, all letters and decimal numbers """
    if ch.isspace():
        return " "
    category = unicodedata.category(ch)
    if category.startswith("L"):
        return ch
    if category.startswith("Nd"):
        return ch
    return " "


def normalize(text: str) -> str:
    """
    Basic normalization step for text transcripts
    NFC normalization, lowercase, removal punctuation/symbols
    """
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = "".join(_replaceCharacter(ch) for ch in text)
    text = _WHITESPACE_PATTERN.sub(" ",  text)
    text = text.strip()

    return text
