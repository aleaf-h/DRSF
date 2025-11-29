from .network import *
from .common import *

MODELS = {
    "DRSF": DRSF
}


def load_model(name="DRSF"):
    assert name in MODELS.keys(), f"Model name can only be one of {MODELS.keys()}."
    print(f"Using model: '{name}'")
    return MODELS[name]
