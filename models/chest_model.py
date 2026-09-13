"""Public entry point for the chest model. The real implementation lives in
custom_architectures.py; this module exists to match the project's file
layout (main.py and other callers import from here or from `models`
directly, both work).
"""
from models.custom_architectures import ChestMedicalNet

__all__ = ["ChestMedicalNet"]
