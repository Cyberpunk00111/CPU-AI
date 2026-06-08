"""Model components for the CogniCore CPU-native SLM.

Imports are intentionally light here so configuration tools can run before
optional heavy dependencies such as PyTorch are installed.
"""

__all__ = ["architecture", "attention", "bitlinear", "config", "ffn", "inference", "rmsnorm"]
