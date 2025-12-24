import gradio as gr
import os
import tempfile
import logging

logger = logging.getLogger(__name__)

def generate_speech(model_name: str, text: str) -> str:
    """
    Generates speech using the selected model.
    
    Args:
        model_name: The name of the model to use (Kokoro, Kitten).
        text: The text to synthesize.
        
    Returns:
        The path to the generated audio file.
    """
    if not text.strip():
        raise gr.Error("Please enter some text to synthesize.")
        
    output_path = tempfile.mktemp(suffix=".wav")
    
    try:
        if model_name == "Kokoro":
            from src.voice_cloning.tts.kokoro import synthesize_speech as kokoro_synthesize
            kokoro_synthesize(text=text, output_path=output_path)
            
        elif model_name == "Kitten":
            from src.voice_cloning.tts.kitten_nano import KittenNanoTTS
            # Instantiate model (lazy loading)
            # TODO: Cache model instance to avoid reloading every time
            tts = KittenNanoTTS()
            tts.synthesize_to_file(text=text, output_path=output_path)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        return output_path
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise gr.Error(f"Synthesis failed: {str(e)}")

def create_tts_tab():
    """Creates the TTS tab content."""
    with gr.Column() as tts_layout:
        gr.Markdown("## Text-to-Speech")
        
        with gr.Row():
            with gr.Column():
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=["Kokoro", "Kitten"],
                    value="Kokoro",
                    interactive=True
                )
                text_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text to synthesize...",
                    lines=3
                )
                generate_btn = gr.Button("Generate Speech", variant="primary")
            
            with gr.Column():
                audio_output = gr.Audio(label="Generated Audio", type="filepath")
        
        # Event wiring
        generate_btn.click(
            fn=generate_speech,
            inputs=[model_dropdown, text_input],
            outputs=[audio_output]
        )
        
    return tts_layout
