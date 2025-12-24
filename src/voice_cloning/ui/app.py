import gradio as gr
from src.voice_cloning.ui.tts_tab import create_tts_tab
from src.voice_cloning.ui.asr_tab import create_asr_tab
from src.voice_cloning.ui.vad_tab import create_vad_tab

def create_interface() -> gr.Blocks:
    """Creates the main Gradio interface for the Voice Cloning Toolkit."""
    with gr.Blocks(title="Voice Cloning Toolkit") as demo:
        gr.Markdown("# Voice Cloning Toolkit")
        gr.Markdown("Select a tab below to access different features.")
        
        with gr.Tabs():
            with gr.TabItem("TTS"):
                create_tts_tab()
            
            with gr.TabItem("ASR"):
                create_asr_tab()
                
            with gr.TabItem("VAD"):
                create_vad_tab()
                
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
