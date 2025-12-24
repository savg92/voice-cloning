import gradio as gr
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

def transcribe_speech(
    model_name: str, 
    audio_path: str,
    whisper_model_id: str,
    whisper_lang: str,
    whisper_task: str,
    whisper_use_mlx: bool,
    whisper_timestamps: bool,
    parakeet_timestamps: bool,
    canary_source_lang: str,
    canary_target_lang: str
) -> str:
    """
    Transcribes speech using the selected model and its parameters.
    """
    if not audio_path:
        raise gr.Error("Please upload an audio file to transcribe.")
        
    try:
        if model_name == "Whisper":
            from src.voice_cloning.asr.whisper import WhisperASR
            model = WhisperASR(model_id=whisper_model_id, use_mlx=whisper_use_mlx)
            lang = whisper_lang if whisper_lang != "auto" else None
            return model.transcribe(audio_path, lang=lang, task=whisper_task, timestamps=whisper_timestamps)
            
        elif model_name == "Parakeet":
            from src.voice_cloning.asr.parakeet import ParakeetASR
            model = ParakeetASR()
            return model.transcribe(audio_path, timestamps=parakeet_timestamps)
            
        elif model_name == "Canary":
            from src.voice_cloning.asr.canary import CanaryASR
            model = CanaryASR()
            model.load_model()
            result = model.transcribe(
                audio_path=audio_path,
                source_lang=canary_source_lang,
                target_lang=canary_target_lang
            )
            return result['text']
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise gr.Error(f"Transcription failed: {str(e)}")

def on_model_change(model_name: str):
    """Updates visibility of model-specific parameter groups."""
    return [
        gr.update(visible=(model_name == "Whisper")),
        gr.update(visible=(model_name == "Parakeet")),
        gr.update(visible=(model_name == "Canary"))
    ]

def create_asr_tab():
    """Creates the ASR tab content."""
    with gr.Column() as asr_layout:
        gr.Markdown("## 📝 Automatic Speech Recognition & Translation")
        
        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    label="ASR Engine",
                    choices=["Whisper", "Parakeet", "Canary"],
                    value="Whisper",
                    interactive=True
                )
                
                audio_input = gr.Audio(
                    label="Source Audio",
                    type="filepath"
                )
                
                # --- Whisper Parameters ---
                with gr.Group(visible=True) as whisper_params:
                    gr.Markdown("### ⚙️ Whisper Settings")
                    whisper_model_id = gr.Dropdown(
                        label="Model Version",
                        choices=[
                            "openai/whisper-large-v3-turbo",
                            "openai/whisper-large-v3",
                            "openai/whisper-medium",
                            "openai/whisper-base",
                            "openai/whisper-tiny",
                            "mlx-community/whisper-large-v3-turbo",
                            "mlx-community/whisper-medium"
                        ],
                        value="openai/whisper-large-v3-turbo"
                    )
                    whisper_lang = gr.Textbox(label="Language (e.g. 'en', 'fr')", value="auto", info="Type 'auto' for detection.")
                    whisper_task = gr.Radio(
                        label="Task Type",
                        choices=["transcribe", "translate"],
                        value="transcribe",
                        info="'translate' translates to English."
                    )
                    with gr.Row():
                        whisper_use_mlx = gr.Checkbox(label="Use MLX (Apple Silicon)", value=True)
                        whisper_timestamps = gr.Checkbox(label="Word Timestamps", value=True)

                # --- Parakeet Parameters ---
                with gr.Group(visible=False) as parakeet_params:
                    gr.Markdown("### ⚙️ Parakeet Settings")
                    parakeet_timestamps = gr.Checkbox(label="Enable SRT Timestamps", value=False)

                # --- Canary Parameters ---
                with gr.Group(visible=False) as canary_params:
                    gr.Markdown("### ⚙️ Canary Settings")
                    canary_source_lang = gr.Dropdown(
                        label="Source Language",
                        choices=['en', 'de', 'fr', 'es', 'it', 'nl', 'pt', 'ru'],
                        value="en"
                    )
                    canary_target_lang = gr.Dropdown(
                        label="Target Language",
                        choices=['en', 'de', 'fr', 'es', 'it', 'nl', 'pt', 'ru'],
                        value="en"
                    )

                transcribe_btn = gr.Button("✨ Transcribe Audio", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("### Result")
                transcript_output = gr.Textbox(
                    label="Transcript",
                    placeholder="Result will appear here...",
                    lines=25,
                    show_copy_button=True
                )
        
        # UI logic for parameter visibility
        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[whisper_params, parakeet_params, canary_params]
        )
        
        # Event wiring
        transcribe_btn.click(
            fn=transcribe_speech,
            inputs=[
                model_dropdown, 
                audio_input,
                whisper_model_id,
                whisper_lang,
                whisper_task,
                whisper_use_mlx,
                whisper_timestamps,
                parakeet_timestamps,
                canary_source_lang,
                canary_target_lang
            ],
            outputs=[transcript_output]
        )
        
    return asr_layout