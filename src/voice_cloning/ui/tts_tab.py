import gradio as gr
import os
import tempfile
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def generate_speech(model_name: str, text: str, reference_audio: Optional[str] = None) -> str:
    """
    Generates speech using the selected model.
    
    Args:
        model_name: The name of the model to use.
        text: The text to synthesize.
        reference_audio: Optional path to reference audio for voice cloning.
        
    Returns:
        The path to the generated audio file.
    """
    if not text.strip():
        raise gr.Error("Please enter some text to synthesize.")
        
    # Validation for cloning models
    cloning_models = ["Chatterbox", "Marvis"]
    if model_name in cloning_models and not reference_audio:
        raise gr.Error(f"Reference audio is required for model '{model_name}'.")

    output_path = tempfile.mktemp(suffix=".wav")
    
    try:
        if model_name == "Kokoro":
            from src.voice_cloning.tts.kokoro import synthesize_speech as kokoro_synthesize
            kokoro_synthesize(text=text, output_path=output_path)
            
        elif model_name == "Kitten":
            from src.voice_cloning.tts.kitten_nano import KittenNanoTTS
            tts = KittenNanoTTS()
            tts.synthesize_to_file(text=text, output_path=output_path)
            
        elif model_name == "Chatterbox":
            from src.voice_cloning.tts.chatterbox import synthesize_with_chatterbox
            synthesize_with_chatterbox(
                text=text,
                output_wav=output_path,
                source_wav=reference_audio
            )
            
        elif model_name == "Marvis":
            from src.voice_cloning.tts.marvis import MarvisTTS
            tts = MarvisTTS()
            tts.synthesize(text=text, output_path=output_path, ref_audio=reference_audio)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        return output_path
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise gr.Error(f"Synthesis failed: {str(e)}")

def on_model_change(model_name: str):
    """Updates UI components based on the selected model."""
    cloning_models = ["Chatterbox", "Marvis"]
    is_cloning = model_name in cloning_models
    return gr.update(visible=is_cloning)

def create_tts_tab():
    """Creates the TTS tab content."""
    with gr.Column() as tts_layout:
        gr.Markdown("## Text-to-Speech")
        
        with gr.Row():
            with gr.Column():
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=["Kokoro", "Kitten", "Chatterbox", "Marvis"],
                    value="Kokoro",
                    interactive=True
                )
                text_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text to synthesize...",
                    lines=3
                )
                
                # Reference Audio Input (hidden by default)
                ref_audio_input = gr.Audio(
                    label="Reference Audio (Voice Cloning)",
                    type="filepath",
                    visible=False
                )
                
                generate_btn = gr.Button("Generate Speech", variant="primary")
            
            with gr.Column():
                audio_output = gr.Audio(label="Generated Audio", type="filepath")
        
        # UI Logic
        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[ref_audio_input]
        )
        
        # Event wiring
        generate_btn.click(
            fn=generate_speech,
            inputs=[model_dropdown, text_input, ref_audio_input],
            outputs=[audio_output]
        )
        
    return tts_layout