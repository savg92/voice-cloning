import pytest

def test_ui_package_exists():
    """Test that the voice_cloning.ui package exists and can be imported."""
    try:
        import src.voice_cloning.ui  # noqa: F401
    except ImportError:
        pytest.fail("Could not import src.voice_cloning.ui package")

def test_ui_package_is_package():
    """Test that voice_cloning.ui is actually a package (has __path__)."""
    import src.voice_cloning.ui
    assert hasattr(src.voice_cloning.ui, "__path__")
