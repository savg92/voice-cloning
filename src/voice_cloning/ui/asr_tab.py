import gradio as gr
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def transcribe_speech(model_name: str, audio_path: str) -> str:
    """
    Transcribes speech using the selected model.
    
    Args:
        model_name: The name of the model to use.
        audio_path: The path to the audio file.
        
    Returns:
        The transcription text.
    """
    if not audio_path:
        raise gr.Error("Please upload an audio file to transcribe.")
        
    try:
        if model_name == "Whisper":
            from src.voice_cloning.asr.whisper import WhisperASR
            model = WhisperASR()
            return model.transcribe(audio_path)
            
        elif model_name == "Parakeet":
            from src.voice_cloning.asr.parakeet import ParakeetASR
            model = ParakeetASR()
            return model.transcribe(audio_path)
            
        elif model_name == "Canary":
            from src.voice_cloning.asr.canary import CanaryASR
            model = CanaryASR()
            model.load_model()
            result = model.transcribe(audio_path)
            return result['text']
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise gr.Error(f"Transcription failed: {str(e)}")

def create_asr_tab():
    """Creates the ASR tab content."""
    with gr.Column() as asr_layout:
        gr.Markdown("## Automatic Speech Recognition")
        
        with gr.Row():
            with gr.Column():
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=["Whisper", "Parakeet", "Canary"],
                    value="Whisper",
                    interactive=True
                )
                audio_input = gr.Audio(
                    label="Upload Audio",
                    type="filepath"
                )
                transcribe_btn = gr.Button("Transcribe", variant="primary")
            
            with gr.Column():
                transcript_output = gr.Textbox(
                    label="Transcription",
                    placeholder="Transcription will appear here...",
                    lines=10
                )
        
        # Event wiring
        transcribe_btn.click(
            fn=transcribe_speech,
            inputs=[model_dropdown, audio_input],
            outputs=[transcript_output]
        )
        
    return asr_layout