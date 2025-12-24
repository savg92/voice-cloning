import gradio as gr
from src.voice_cloning.ui.tts_tab import create_tts_tab
from src.voice_cloning.ui.asr_tab import create_asr_tab
from src.voice_cloning.ui.vad_tab import create_vad_tab

def create_interface() -> gr.Blocks:
    """Creates the main Gradio interface for the Voice Cloning Toolkit."""
    with gr.Blocks(title="Voice Cloning Toolkit", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎙️ Voice Cloning & ASR Research Toolkit")
        gr.Markdown(
            "Welcome to the Voice Cloning Research Toolkit. This interface allows you to "
            "interactively test and compare state-of-the-art TTS, ASR, and VAD models."
        )
        
        with gr.Tabs():
            with gr.TabItem("🎭 Text-to-Speech"):
                create_tts_tab()
            
            with gr.TabItem("📝 Speech Recognition"):
                create_asr_tab()
                
            with gr.TabItem("🔍 Voice Activity"):
                create_vad_tab()
                
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
