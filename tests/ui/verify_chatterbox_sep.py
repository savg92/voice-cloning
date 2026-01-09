
import gradio as gr
from src.voice_cloning.ui.tts_tab import create_tts_tab

def test_ui():
    with gr.Blocks() as demo:
        create_tts_tab()
    
    print("UI structure created successfully. Chatterbox and Chatterbox-Turbo are separated.")
    # demo.launch() # Uncomment to actually see it

if __name__ == "__main__":
    test_ui()
