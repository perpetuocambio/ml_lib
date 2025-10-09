"""Accept-Language header values as enum."""

from enum import Enum


class AcceptLanguageValue(Enum):
    """Accept-Language header values."""

    ENGLISH_US = "en-US,en;q=0.5"
    SPANISH = "es-ES,es;q=0.8"
    FRENCH = "fr-FR,fr;q=0.8"
    GERMAN = "de-DE,de;q=0.8"
