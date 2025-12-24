import pytest
from unittest.mock import patch, MagicMock
import sys
from main import main

@patch("src.voice_cloning.ui.app.create_interface")
@patch("sys.argv", ["main.py", "--model", "web"])
def test_main_web_mode(mock_create_interface):
    """Test that --model web launches the Gradio interface."""
    mock_demo = MagicMock()
    mock_create_interface.return_value = mock_demo
    
    # We need to catch SystemExit because main() might exit if I don't implement it right
    # But if implemented right, it should run web and exit normally (or block, but launch() blocks).
    # launch() is mocked.
    
    try:
        main()
    except SystemExit as e:
        # If it exits with 0, it's fine? Or maybe it shouldn't exit.
        if e.code != 0:
            pytest.fail(f"main() exited with code {e.code}")
            
    mock_create_interface.assert_called_once()
    mock_demo.launch.assert_called_once()
