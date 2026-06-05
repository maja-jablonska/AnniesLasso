import os

# Pin the JAX CPU backend for the test session *before* anything imports jax.
# This root conftest is loaded by pytest before the `thecannon` package (and
# therefore jax) is imported, which the in-package tests/conftest.py cannot
# guarantee. Users with a working accelerator can override JAX_PLATFORMS.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
