import os

# Run the test suite on the CPU backend for determinism and portability. This
# must be set before JAX initializes its backend. Users with a working
# accelerator can override by exporting JAX_PLATFORMS themselves.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
