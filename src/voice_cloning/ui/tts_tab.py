import gradio as gr
import os
import tempfile
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def generate_speech(
    model_name: str, 
    text: str, 
    reference_audio: Optional[str],
    # Shared params
    speed: float,
    use_mlx: bool,
    # Kokoro params
    kokoro_voice: str,
    kokoro_lang: str,
    # Kitten params
    kitten_voice: str,
    # Chatterbox params
    chatter_exaggeration: float,
    chatter_cfg: float,
    chatter_lang: str,
    chatter_multi: bool,
    # Marvis params
    marvis_temp: float,
    marvis_top_p: float,
    marvis_quant: bool,
    # CosyVoice params
    cosy_instruct: str
) -> str:
    """
    Generates speech using the selected model and its specific parameters.
    """
    if not text.strip():
        raise gr.Error("Please enter some text to synthesize.")
        
    # Validation for cloning models
    cloning_models = ["Chatterbox", "Marvis", "CosyVoice"]
    if model_name in cloning_models and not reference_audio:
        raise gr.Error(f"Reference audio is required for model '{model_name}'.")

    output_path = tempfile.mktemp(suffix=".wav")
    
    try:
        if model_name == "Kokoro":
            from src.voice_cloning.tts.kokoro import synthesize_speech as kokoro_synthesize
            kokoro_synthesize(
                text=text, 
                output_path=output_path, 
                voice=kokoro_voice,
                lang_code=kokoro_lang,
                speed=speed,
                use_mlx=use_mlx
            )
            
        elif model_name == "Kitten":
            from src.voice_cloning.tts.kitten_nano import KittenNanoTTS
            tts = KittenNanoTTS()
            tts.synthesize_to_file(
                text=text, 
                output_path=output_path,
                voice=kitten_voice,
                speed=speed
            )
            
        elif model_name == "Chatterbox":
            from src.voice_cloning.tts.chatterbox import synthesize_with_chatterbox
            synthesize_with_chatterbox(
                text=text,
                output_wav=output_path,
                source_wav=reference_audio,
                exaggeration=chatter_exaggeration,
                cfg_weight=chatter_cfg,
                language=chatter_lang,
                multilingual=chatter_multi,
                use_mlx=use_mlx
            )
            
        elif model_name == "Marvis":
            from src.voice_cloning.tts.marvis import MarvisTTS
            tts = MarvisTTS()
            tts.synthesize(
                text=text, 
                output_path=output_path, 
                ref_audio=reference_audio,
                speed=speed,
                temperature=marvis_temp,
                top_p=marvis_top_p,
                quantized=marvis_quant
            )
            
        elif model_name == "CosyVoice":
            from src.voice_cloning.tts.cosyvoice import synthesize_speech as cosy_synthesize
            cosy_synthesize(
                text=text,
                output_path=output_path,
                ref_audio_path=reference_audio,
                instruct_text=cosy_instruct,
                speed=speed,
                use_mlx=use_mlx
            )
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        return output_path
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise gr.Error(f"Synthesis failed: {str(e)}")

def on_model_change(model_name: str):
    """Updates visibility of model-specific parameter groups."""
    cloning_models = ["Chatterbox", "Marvis", "CosyVoice"]
    is_cloning = model_name in cloning_models
    
    return [
        gr.update(visible=is_cloning),           # ref_audio_input
        gr.update(visible=(model_name == "Kokoro")),    # kokoro_params
        gr.update(visible=(model_name == "Kitten")),    # kitten_params
        gr.update(visible=(model_name == "Chatterbox")),# chatter_params
        gr.update(visible=(model_name == "Marvis")),    # marvis_params
        gr.update(visible=(model_name == "CosyVoice"))  # cosy_params
    ]

def create_tts_tab():
    """Creates the TTS tab content."""
    with gr.Column() as tts_layout:
        gr.Markdown("## Text-to-Speech")
        
        with gr.Row():
            with gr.Column():
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=["Kokoro", "Kitten", "Chatterbox", "Marvis", "CosyVoice"],
                    value="Kokoro",
                    interactive=True
                )
                text_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text to synthesize...",
                    lines=3
                )
                
                # Reference Audio Input
                ref_audio_input = gr.Audio(
                    label="Reference Audio (Voice Cloning)",
                    type="filepath",
                    visible=False
                )

                # --- Shared Params ---
                with gr.Row():
                    speed = gr.Slider(label="Speed", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
                    use_mlx = gr.Checkbox(label="Use MLX (Apple Silicon)", value=True)

                # --- Kokoro Params ---
                with gr.Group(visible=True) as kokoro_params:
                    gr.Markdown("### Kokoro Settings")
                    kokoro_voice = gr.Dropdown(
                        label="Voice",
                        choices=["af_heart", "af_bella", "am_adam", "bf_emma", "bf_isabella"],
                        value="af_heart"
                    )
                    kokoro_lang = gr.Dropdown(label="Language Code", choices=["a", "b", "e"], value="a")

                # --- Kitten Params ---
                with gr.Group(visible=False) as kitten_params:
                    gr.Markdown("### Kitten Settings")
                    kitten_voice = gr.Dropdown(
                        label="Voice",
                        choices=["expr-voice-4-f", "expr-voice-1-m"],
                        value="expr-voice-4-f"
                    )

                # --- Chatterbox Params ---
                with gr.Group(visible=False) as chatter_params:
                    gr.Markdown("### Chatterbox Settings")
                    chatter_exaggeration = gr.Slider(label="Exaggeration", minimum=0.0, maximum=1.0, value=0.7)
                    chatter_cfg = gr.Slider(label="CFG Weight", minimum=0.0, maximum=1.0, value=0.5)
                    chatter_lang = gr.Textbox(label="Language Code", value="en")
                    chatter_multi = gr.Checkbox(label="Multilingual Mode", value=False)

                # --- Marvis Params ---
                with gr.Group(visible=False) as marvis_params:
                    gr.Markdown("### Marvis Settings")
                    marvis_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.7)
                    marvis_top_p = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, value=0.95)
                    marvis_quant = gr.Checkbox(label="Quantized (4-bit)", value=True)

                # --- CosyVoice Params ---
                with gr.Group(visible=False) as cosy_params:
                    gr.Markdown("### CosyVoice Settings")
                    cosy_instruct = gr.Textbox(label="Instruct Text (Style)", placeholder="e.g. 'Speak with excitement'")

                generate_btn = gr.Button("Generate Speech", variant="primary")
            
            with gr.Column():
                audio_output = gr.Audio(label="Generated Audio", type="filepath")
        
        # UI Logic
        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[
                ref_audio_input, 
                kokoro_params, 
                kitten_params, 
                chatter_params, 
                marvis_params, 
                cosy_params
            ]
        )
        
        # Event wiring
        generate_btn.click(
            fn=generate_speech,
            inputs=[
                model_dropdown, 
                text_input, 
                ref_audio_input,
                speed,
                use_mlx,
                kokoro_voice,
                kokoro_lang,
                kitten_voice,
                chatter_exaggeration,
                chatter_cfg,
                chatter_lang,
                chatter_multi,
                marvis_temp,
                marvis_top_p,
                marvis_quant,
                cosy_instruct
            ],
            outputs=[audio_output]
        )
        
    return tts_layout