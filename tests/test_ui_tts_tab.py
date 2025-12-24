import pytest
import gradio as gr
from src.voice_cloning.ui.tts_tab import create_tts_tab

def test_create_tts_tab_returns_component():
    """Test that create_tts_tab returns a Gradio Component (likely a Column or Group)."""
    with gr.Blocks():
        tab = create_tts_tab()
        # It usually returns a Container/Column/Group which inherits from BlockContext/Component
        assert isinstance(tab, gr.blocks.BlockContext) or isinstance(tab, gr.components.Component)
