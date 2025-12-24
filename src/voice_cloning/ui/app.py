import gradio as gr
from src.voice_cloning.ui.tts_tab import create_tts_tab

def create_interface() -> gr.Blocks:
    """Creates the main Gradio interface for the Voice Cloning Toolkit."""
    with gr.Blocks(title="Voice Cloning Toolkit") as demo:
        gr.Markdown("# Voice Cloning Toolkit")
        gr.Markdown("Select a tab below to access different features.")
        
        with gr.Tabs():
            with gr.TabItem("TTS"):
                create_tts_tab()
            
            with gr.TabItem("ASR"):
                gr.Markdown("## Automatic Speech Recognition")
                gr.Markdown("Coming soon...")
                
            with gr.TabItem("VAD"):
                gr.Markdown("## Voice Activity Detection")
                gr.Markdown("Coming soon...")
                
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
