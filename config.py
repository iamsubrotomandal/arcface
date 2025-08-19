"""Central configuration for model weight paths.

Environment override:
  ARCFACE_WEIGHTS -> path to arcface iresnet100 weights (.pth)
"""
import os
from pathlib import Path

default_arcface_weights = 'weights/arcface_iresnet100_auto.pth'

def get_arcface_weight_path():
    env_path = os.environ.get('ARCFACE_WEIGHTS')
    if env_path and os.path.isfile(env_path):
        return env_path
    if os.path.isfile(default_arcface_weights):
        return default_arcface_weights
    return None

__all__ = ['get_arcface_weight_path']
