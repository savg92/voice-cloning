import pytest
import gradio as gr
from src.voice_cloning.ui.asr_tab import create_asr_tab

def test_create_asr_tab_returns_component():
    """Test that create_asr_tab returns a Gradio Component."""
    with gr.Blocks():
        tab = create_asr_tab()
        assert isinstance(tab, gr.blocks.BlockContext) or isinstance(tab, gr.components.Component)
