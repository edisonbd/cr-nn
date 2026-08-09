"""Backend implementations.

Submodules are imported lazily from :mod:`crnn.backend` so that a machine
without MLX never imports it (and vice versa for torch).
"""
