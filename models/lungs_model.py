"""Public entry point for the lungs model. See custom_architectures.py's
LungsMedicalNet docstring: this is a structural skeleton only, not a
trained or trainable model, pending lung CT data acquisition.
"""
from models.custom_architectures import LungsMedicalNet

__all__ = ["LungsMedicalNet"]
