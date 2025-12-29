import gradio as gr
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.voice_cloning.ui.app import create_interface

def test_create_interface_returns_blocks():
    """Test that create_interface returns a Gradio Blocks instance."""
    demo = create_interface()
    assert isinstance(demo, gr.Blocks)

def test_interface_title():
    """Test that the interface has the correct title."""
    demo = create_interface()
    # Gradio Blocks title is stored in the .title attribute
    # Note: Depending on Gradio version, access might differ, but .title is standard.
    assert "Voice Cloning Toolkit" in demo.title
