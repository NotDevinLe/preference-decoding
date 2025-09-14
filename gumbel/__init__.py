"""
Gumbel preference decoding package.
"""

__version__ = "0.1.0"

# Make core components available at package level
from . import core
from . import utils
from . import tests

__all__ = [
    "core",
    "utils", 
    "tests",
]