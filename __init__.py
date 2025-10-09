"""
__init__.py para el paquete ml_lib
"""

from . import core
from . import linalg
from . import visualization

# Asegurarse de que los submódulos estén disponibles
__all__ = ["core", "linalg", "visualization"]
