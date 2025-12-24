import gradio as gr
import os
import tempfile
import logging
import torch
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# Kokoro voice mapping (Expanded)
KOKORO_VOICES = {
    "a": ["af_heart", "af_bella", "am_adam", "am_fenix", "am_puck", "af_nicole", "af_sky"],
    "b": ["bf_emma", "bf_isabella", "bm_george", "bm_lewis"],
    "d": ["de_sarah"],
    "e": ["ef_dora"],
    "f": ["ff_siwis"],
    "h": ["hf_alpha", "hi_puck"],
    "i": ["if_sara", "it_bella"],
    "j": ["jf_alpha"],
    "p": ["pf_dora"],
    "r": ["ru_nicole"],
    "t": ["tr_river"],
    "z": ["zf_alpha"]
}

KOKORO_LANGS = {
    "a": "US English",
    "b": "British English",
    "d": "German",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Portuguese",
    "r": "Russian",
    "t": "Turkish",
    "z": "Chinese"
}

def generate_speech(
    model_name: str, 
    text: str, 
    reference_audio: Optional[str],
    reference_text: Optional[str],
    speed: float,
    use_mlx: bool,
    stream: bool,
    # Kokoro
    kokoro_lang: str,
    kokoro_voice: str,
    # Kitten
    kitten_voice: str,
    # Chatterbox
    chatter_exaggeration: float,
    chatter_cfg: float,
    chatter_lang: str,
    chatter_multi: bool,
    chatter_model_id: str,
    chatter_voice: str,
    # Marvis
    marvis_temp: float,
    marvis_top_p: float,
    marvis_quant: bool,
    # CosyVoice
    cosy_instruct: str,
    # NeuTTS Air
    neutts_backbone: str,
    # Supertone
    supertone_preset: str,
    supertone_steps: int,
    supertone_cfg: float,
    # Dia2
    dia2_cfg: float,
    dia2_temp: float,
    dia2_top_k: int
) -> str:
    """
    Generates speech using the selected model and ALL available parameters.
    """
    if not text.strip():
        raise gr.Error("Please enter some text to synthesize.")
        
    cloning_models = ["Chatterbox", "Marvis", "CosyVoice", "NeuTTS Air", "OpenVoice", "OpenVoice2"]
    if model_name in cloning_models and not reference_audio:
        raise gr.Error(f"Reference audio is required for model '{model_name}'.")

    output_path = tempfile.mktemp(suffix=".wav")
    
    try:
        gr.Info(f"Synthesizing with {model_name}...")
        
        if model_name == "Kokoro":
            from src.voice_cloning.tts.kokoro import synthesize_speech as kokoro_synthesize
            kokoro_synthesize(
                text=text, output_path=output_path, voice=kokoro_voice,
                lang_code=kokoro_lang, speed=speed, use_mlx=use_mlx, stream=stream
            )
            
        elif model_name == "Kitten":
            from src.voice_cloning.tts.kitten_nano import KittenNanoTTS
            tts = KittenNanoTTS()
            tts.synthesize_to_file(
                text=text, output_path=output_path, voice=kitten_voice, speed=speed, stream=stream
            )
            
        elif model_name == "Chatterbox":
            from src.voice_cloning.tts.chatterbox import synthesize_with_chatterbox
            synthesize_with_chatterbox(
                text=text, output_wav=output_path, source_wav=reference_audio,
                exaggeration=chatter_exaggeration, cfg_weight=chatter_cfg,
                language=chatter_lang, multilingual=chatter_multi, use_mlx=use_mlx,
                model_id=chatter_model_id if chatter_model_id else None,
                voice=chatter_voice if chatter_voice else None
            )
            
        elif model_name == "Marvis":
            from src.voice_cloning.tts.marvis import MarvisTTS
            tts = MarvisTTS()
            tts.synthesize(
                text=text, output_path=output_path, ref_audio=reference_audio,
                ref_text=reference_text, speed=speed, temperature=marvis_temp,
                top_p=marvis_top_p, quantized=marvis_quant, stream=stream
            )
            
        elif model_name == "CosyVoice":
            from src.voice_cloning.tts.cosyvoice import synthesize_speech as cosy_synthesize
            cosy_synthesize(
                text=text, output_path=output_path, ref_audio_path=reference_audio,
                ref_text=reference_text, instruct_text=cosy_instruct,
                speed=speed, use_mlx=use_mlx
            )

        elif model_name == "NeuTTS Air":
            from src.voice_cloning.tts.neutts_air import synthesize_with_neutts_air
            synthesize_with_neutts_air(
                text=text, output_path=output_path, ref_audio=reference_audio,
                ref_text=reference_text, backbone=neutts_backbone, device="cpu"
            )

        elif model_name == "Supertone":
            from src.voice_cloning.tts.supertone import synthesize_with_supertone
            synthesize_with_supertone(
                text=text, output_path=output_path, preset=supertone_preset,
                steps=supertone_steps, cfg_scale=supertone_cfg, stream=stream
            )

        elif model_name == "Dia2":
            if not torch.cuda.is_available():
                raise gr.Error("Dia2 model requires an NVIDIA GPU with CUDA. CUDA was not detected.")
            try:
                from src.voice_cloning.tts.dia2 import Dia2TTS
                tts = Dia2TTS()
                tts.synthesize(
                    text=text, output_path=output_path, cfg_scale=dia2_cfg,
                    temperature=dia2_temp, top_k=dia2_top_k
                )
            except ImportError:
                raise gr.Error("Dia2 library not found. Please install it to use this model.")
        
        elif model_name in ["OpenVoice", "OpenVoice2"]:
            # Note: OpenVoice is mentioned in PRD but not yet implemented in tts/
            raise gr.Error(f"{model_name} is currently not implemented in the backend.")
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        gr.Info("Synthesis complete!")
        return output_path
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise gr.Error(f"Synthesis failed: {str(e)}")

def on_model_change(model_name: str):
    """Updates visibility of model-specific parameter groups."""
    cloning_models = ["Chatterbox", "Marvis", "CosyVoice", "NeuTTS Air", "OpenVoice", "OpenVoice2"]
    ref_text_models = ["Marvis", "CosyVoice", "NeuTTS Air"]
    
    return [
        gr.update(visible=(model_name in cloning_models)), # ref_audio
        gr.update(visible=(model_name in ref_text_models)), # ref_text
        gr.update(visible=(model_name == "Kokoro")),
        gr.update(visible=(model_name == "Kitten")),
        gr.update(visible=(model_name == "Chatterbox")),
        gr.update(visible=(model_name == "Marvis")),
        gr.update(visible=(model_name == "CosyVoice")),
        gr.update(visible=(model_name == "NeuTTS Air")),
        gr.update(visible=(model_name == "Supertone")),
        gr.update(visible=(model_name == "Dia2"))
    ]

def on_kokoro_lang_change(lang_code: str):
    """Updates Kokoro voices based on the selected language."""
    voices = KOKORO_VOICES.get(lang_code, ["af_heart"])
    return gr.update(choices=voices, value=voices[0])

def create_tts_tab():
    """Creates the TTS tab content."""
    with gr.Column() as tts_layout:
        gr.Markdown("## 🎭 Text-to-Speech & Voice Cloning")
        
        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    label="Model Engine",
                    choices=["Kokoro", "Kitten", "Chatterbox", "Marvis", "CosyVoice", "NeuTTS Air", "Supertone", "Dia2", "OpenVoice", "OpenVoice2"],
                    value="Kokoro",
                    interactive=True
                )
                
                text_input = gr.Textbox(label="Input Text", placeholder="Text to synthesize...", lines=4)
                
                ref_audio_input = gr.Audio(label="Reference Audio (Cloning)", type="filepath", visible=False)
                ref_text_input = gr.Textbox(label="Reference Text (Optional transcript)", visible=False)

                with gr.Accordion("⚙️ Global Settings", open=True):
                    with gr.Row():
                        speed = gr.Slider(label="Speed", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
                        use_mlx = gr.Checkbox(label="MLX Acceleration", value=True)
                        stream = gr.Checkbox(label="Enable Streaming", value=False)

                # --- Model Groups ---
                with gr.Group(visible=True) as kokoro_params:
                    gr.Markdown("### Kokoro Settings")
                    with gr.Row():
                        kokoro_lang = gr.Dropdown(
                            label="Language", 
                            choices=[(v, k) for k, v in KOKORO_LANGS.items()], 
                            value="a"
                        )
                        kokoro_voice = gr.Dropdown(
                            label="Voice", 
                            choices=KOKORO_VOICES["a"], 
                            value="af_heart"
                        )

                with gr.Group(visible=False) as kitten_params:
                    gr.Markdown("### Kitten Settings")
                    kitten_voice = gr.Dropdown(label="Voice", choices=["expr-voice-4-f", "expr-voice-1-m"], value="expr-voice-4-f")

                with gr.Group(visible=False) as chatter_params:
                    gr.Markdown("### Chatterbox Settings")
                    with gr.Row():
                        chatter_exaggeration = gr.Slider(label="Exaggeration", minimum=0.0, maximum=1.0, value=0.7)
                        chatter_cfg = gr.Slider(label="CFG Weight", minimum=0.0, maximum=1.0, value=0.5)
                    with gr.Row():
                        chatter_lang = gr.Textbox(label="Language", value="en")
                        chatter_multi = gr.Checkbox(label="Multilingual Mode", value=False)
                    with gr.Row():
                        chatter_model_id = gr.Textbox(label="Model ID (Optional)")
                        chatter_voice = gr.Textbox(label="Voice Preset (Optional)")

                with gr.Group(visible=False) as marvis_params:
                    gr.Markdown("### Marvis Settings")
                    with gr.Row():
                        marvis_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.7)
                        marvis_top_p = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, value=0.95)
                    marvis_quant = gr.Checkbox(label="4-bit Quantization", value=True)

                with gr.Group(visible=False) as cosy_params:
                    gr.Markdown("### CosyVoice Settings")
                    cosy_instruct = gr.Textbox(label="Style Prompt", placeholder="e.g. 'Speak with excitement'")

                with gr.Group(visible=False) as neutts_params:
                    gr.Markdown("### NeuTTS Air Settings")
                    neutts_backbone = gr.Dropdown(label="Backbone", choices=["neuphonic/neutts-air-q4-gguf", "neuphonic/neutts-air-q8-gguf"], value="neuphonic/neutts-air-q4-gguf")

                with gr.Group(visible=False) as supertone_params:
                    gr.Markdown("### Supertone Settings")
                    supertone_preset = gr.Textbox(label="Voice Preset (e.g. F1, M1)")
                    with gr.Row():
                        supertone_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, value=8, step=1)
                        supertone_cfg = gr.Slider(label="CFG Scale", minimum=0.0, maximum=5.0, value=1.0)

                with gr.Group(visible=False) as dia2_params:
                    gr.Markdown("### Dia2 Settings (CUDA Required)")
                    with gr.Row():
                        dia2_cfg = gr.Slider(label="CFG Scale", minimum=0.0, maximum=5.0, value=2.0)
                        dia2_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.5, value=0.8)
                    dia2_top_k = gr.Slider(label="Top K", minimum=1, maximum=100, value=50, step=1)

                generate_btn = gr.Button("🚀 Generate Audio", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                audio_output = gr.Audio(label="Generated Result", type="filepath")
        
        # UI Logic
        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[
                ref_audio_input, ref_text_input, 
                kokoro_params, kitten_params, chatter_params, 
                marvis_params, cosy_params, neutts_params, 
                supertone_params, dia2_params
            ]
        )
        
        kokoro_lang.change(
            fn=on_kokoro_lang_change,
            inputs=[kokoro_lang],
            outputs=[kokoro_voice]
        )
        
        generate_btn.click(
            fn=generate_speech,
            inputs=[
                model_dropdown, text_input, ref_audio_input, ref_text_input, speed, use_mlx, stream,
                kokoro_lang, kokoro_voice,
                kitten_voice,
                chatter_exaggeration, chatter_cfg, chatter_lang, chatter_multi, chatter_model_id, chatter_voice,
                marvis_temp, marvis_top_p, marvis_quant,
                cosy_instruct,
                neutts_backbone,
                supertone_preset, supertone_steps, supertone_cfg,
                dia2_cfg, dia2_temp, dia2_top_k
            ],
            outputs=[audio_output]
        )
        
    return tts_layout