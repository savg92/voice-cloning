# Expose wrapper names for convenience. Avoid importing heavy modules at package
# import time (torch, transformers, etc.) — keep imports lazy to allow import
# in CI/environments that may not have every optional dependency installed.
__all__ = []


def __getattr__(name: str):
	"""Lazily import sub-tree objects to avoid import-time heavy deps.

	This allows `from src.voice_cloning.tts import KittenNanoTTS` without
	requiring heavy packages to be present during import of the package.
	"""
	if name == "KittenNanoTTS":
		from src.voice_cloning.tts.kitten_nano import KittenNanoTTS

		return KittenNanoTTS
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

