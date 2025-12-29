import pytest

def test_ui_package_exists():
    """Test that the voice_cloning.ui package exists and can be imported."""
    try:
        import voice_cloning.ui  # noqa: F401
    except ImportError:
        pytest.fail("Could not import voice_cloning.ui package")

def test_ui_package_is_package():
    """Test that voice_cloning.ui is actually a package (has __path__)."""
    from voice_cloning import ui
    assert hasattr(ui, "__path__")
