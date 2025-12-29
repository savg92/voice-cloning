import gradio as gr
import logging

logger = logging.getLogger(__name__)

CANARY_LANGS = [
    'bg', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi', 'fr', 'de', 
    'el', 'hu', 'it', 'lv', 'lt', 'mt', 'pl', 'pt', 'ru', 'sk', 
    'sl', 'es', 'sv', 'ru', 'uk'
]

WHISPER_LANGS = [
    "auto", "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "ba", "as", "tt", "ln", "ha", "mg"
]

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
    canary_target_lang: str,
    canary_timestamps: bool
) -> str:
    """
    Transcribes speech using the selected model and its parameters.
    """
    if not audio_path:
        raise gr.Error("Please upload an audio file to transcribe.")
        
    try:
        gr.Info(f"Transcribing with {model_name}...")
        result_text = ""
        
        if model_name == "Whisper":
            from src.voice_cloning.asr.whisper import WhisperASR
            model = WhisperASR(model_id=whisper_model_id, use_mlx=whisper_use_mlx)
            lang = whisper_lang if whisper_lang != "auto" else None
            result_text = model.transcribe(audio_path, lang=lang, task=whisper_task, timestamps=whisper_timestamps)
            
        elif model_name == "Parakeet":
            from src.voice_cloning.asr.parakeet import ParakeetASR
            model = ParakeetASR()
            result_text = model.transcribe(audio_path, timestamps=parakeet_timestamps)
            
        elif model_name == "Canary":
            from src.voice_cloning.asr.canary import CanaryASR
            model = CanaryASR()
            model.load_model()
            result = model.transcribe(
                audio_path=audio_path,
                source_lang=canary_source_lang,
                target_lang=canary_target_lang,
                timestamps=canary_timestamps
            )
            result_text = result['text']
            
            if canary_timestamps and 'timestamps' in result:
                result_text += "\n\n--- Segments ---\n"
                for segment in result['timestamps'].get('segment', []):
                    start = segment.get('start', 0)
                    end = segment.get('end', 0)
                    text = segment.get('segment', '')
                    result_text += f"[{start:.2f}s -> {end:.2f}s] {text}\n"

        elif model_name == "Granite":
            from src.voice_cloning.asr.granite import transcribe_file
            import tempfile
            out_txt = tempfile.mktemp(suffix=".txt")
            transcribe_file(audio_path, out_txt)
            with open(out_txt, "r") as f:
                result_text = f.read()
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        gr.Info("Transcription complete!")
        return result_text
            
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise gr.Error(f"Transcription failed: {str(e)}")

def on_model_change(model_name: str):
    """Updates visibility of model-specific parameter groups."""
    return [
        gr.update(visible=(model_name == "Whisper")),
        gr.update(visible=(model_name == "Parakeet")),
        gr.update(visible=(model_name == "Canary")),
        gr.update(visible=(model_name == "Granite"))
    ]

def create_asr_tab():
    """Creates the ASR tab content."""
    with gr.Column() as asr_layout:
        gr.Markdown("## 📝 Automatic Speech Recognition")
        
        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    label="ASR Engine",
                    choices=["Whisper", "Parakeet", "Canary", "Granite"],
                    value="Whisper",
                    interactive=True
                )
                
                audio_input = gr.Audio(label="Source Audio", type="filepath")
                
                with gr.Group(visible=True) as whisper_params:
                    gr.Markdown("### Whisper Settings")
                    whisper_model_id = gr.Dropdown(
                        label="Model Version",
                        choices=["openai/whisper-large-v3-turbo", "openai/whisper-large-v3", "openai/whisper-medium", "openai/whisper-base", "openai/whisper-tiny", "mlx-community/whisper-large-v3-turbo", "mlx-community/whisper-medium"],
                        value="openai/whisper-large-v3-turbo"
                    )
                    whisper_lang = gr.Dropdown(label="Language", choices=WHISPER_LANGS, value="auto", allow_custom_value=True)
                    whisper_task = gr.Radio(label="Task", choices=["transcribe", "translate"], value="transcribe")
                    whisper_use_mlx = gr.Checkbox(label="Use MLX Acceleration", value=True)
                    whisper_timestamps = gr.Checkbox(label="Show Timestamps", value=True)

                with gr.Group(visible=False) as parakeet_params:
                    gr.Markdown("### Parakeet Settings")
                    parakeet_timestamps = gr.Checkbox(label="Enable SRT Timestamps", value=False)

                with gr.Group(visible=False) as canary_params:
                    gr.Markdown("### Canary Settings")
                    with gr.Row():
                        canary_source_lang = gr.Dropdown(label="Source Language", choices=CANARY_LANGS, value="en")
                        canary_target_lang = gr.Dropdown(label="Target Language", choices=CANARY_LANGS, value="en")
                    canary_timestamps = gr.Checkbox(label="Show Timestamps", value=False)

                with gr.Group(visible=False) as granite_params:
                    gr.Markdown("### Granite Settings")
                    gr.Markdown("Granite ASR currently uses default settings.")

                transcribe_btn = gr.Button("✨ Transcribe Audio", variant="primary")
            
            with gr.Column(scale=1):
                transcript_output = gr.Textbox(label="Transcript", lines=25, show_copy_button=True)
        
        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[whisper_params, parakeet_params, canary_params, granite_params]
        )
        
        transcribe_btn.click(
            fn=transcribe_speech,
            inputs=[
                model_dropdown, audio_input, whisper_model_id, whisper_lang, whisper_task, whisper_use_mlx, whisper_timestamps,
                parakeet_timestamps, canary_source_lang, canary_target_lang, canary_timestamps
            ],
            outputs=[transcript_output]
        )
        
    return asr_layout