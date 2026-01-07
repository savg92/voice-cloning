import gradio as gr
import tempfile
import logging
import torch

logger = logging.getLogger(__name__)

# Kokoro voice mapping (Expanded)
KOKORO_VOICES = {
    "a": ["af_heart", "af_bella", "am_adam", "am_fenix", "am_puck", "af_nicole", "af_sky"],
    "b": ["bf_emma", "bf_isabella", "bm_george", "bm_lewis"],
    "e": ["ef_dora", "em_alex", "em_santa"],
    "f": ["ff_siwis"],
    "h": ["hf_alpha", "hi_puck"],
    "i": ["if_sara", "it_bella", "im_nicola"],
    "j": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "p": ["pf_dora", "pm_alex", "pm_santa"],
    "z": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"]
}

# Chatterbox Constants
try:
    from src.voice_cloning.tts.chatterbox import VOICE_PRESETS
    CHATTERBOX_VOICES = list(VOICE_PRESETS.keys())
except ImportError:
    CHATTERBOX_VOICES = ["af_heart", "ef_dora", "ff_siwis"] # Fallback

CHATTERBOX_LANGS = [
    ("English", "en"),
    ("Arabic", "ar"),
    ("Chinese", "zh"),
    ("Czech", "cs"),
    ("Danish", "da"),
    ("Dutch", "nl"),
    ("Finnish", "fi"),
    ("French", "fr"),
    ("German", "de"),
    ("Greek", "el"),
    ("Hebrew", "he"),
    ("Hindi", "hi"),
    ("Hungarian", "hu"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Malay", "ms"),
    ("Norwegian", "no"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Spanish", "es"),
    ("Swedish", "sv"),
    ("Swahili", "sw"),
    ("Turkish", "tr")
]

CHATTERBOX_MODELS = [
    "mlx-community/chatterbox-4bit",
    "mlx-community/chatterbox-turbo-4bit"
]

MODEL_DESCRIPTIONS = {
    "Kokoro": "⚡ **Fast & High Quality** (82M params). Best overall for English/European languages. Supports streaming.",
    "Kitten": "🐱 **Lightweight** (30M params). Optimized for very low latency.",
    "Chatterbox": "🗣️ **Voice Cloning & Emotion**. Supports 23 languages and zero-shot cloning from audio.",
    "Marvis": "🤖 **Minimalist**. Simple English TTS.",
    "CosyVoice": "🎙️ **Advanced Generation**. Instruct-based control.",
    "NeuTTS Air": "💨 **CPU Optimized**. GGUF based models.",
    "Supertone": "🎛️ **Controllable**. Step-based generation with CFG (v1).",
    "Supertonic-2": "🎙️ **Fast & Multilingual**. New v2 model supporting EN, KO, ES, PT, FR via ONNX.",
    "Dia2": "🎹 **High Fidelity**. Diffusion-based (Requires CUDA).",
}

KOKORO_LANGS = {
    "a": "US English",
    "b": "British English",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Portuguese",
    "z": "Chinese"
}

def generate_speech(
    model_name: str, 
    text: str, 
    reference_audio: str | None,
    reference_text: str | None,
    speed: float,
    use_mlx: bool,
    stream: bool,
    # Kokoro
    kokoro_lang: str,
    kokoro_voice: str,
    # Kitten
    kitten_version: str,
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
    # Supertonic-2
    supertonic2_lang: str,
    supertonic2_voice: str,
    supertonic2_steps: int,
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
        
    cloning_models = ["Chatterbox", "Marvis", "CosyVoice", "NeuTTS Air"]
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
            # Map version name to model ID
            v_map = {
                "0.1": "KittenML/kitten-tts-nano-0.1",
                "0.2": "KittenML/kitten-tts-nano-0.2"
            }
            model_id = v_map.get(kitten_version, "KittenML/kitten-tts-nano-0.2")
            tts = KittenNanoTTS(model_id=model_id)
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
                voice=chatter_voice if chatter_voice else None,
                speed=speed, stream=stream
            )

        elif model_name == "Marvis":
            from src.voice_cloning.tts.marvis import MarvisTTS
            tts = MarvisTTS()
            tts.synthesize(
                text=text, output_path=output_path, ref_audio=reference_audio,
                ref_text=reference_text, speed=speed, temperature=marvis_temp,
                quantized=marvis_quant, stream=stream,
                lang_code="en" # Default to en for Marvis in UI for now
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

        elif model_name == "Supertonic-2":
            from src.voice_cloning.tts.supertonic2 import Supertonic2TTS
            # Note: We use CPU if torch isn't available or if on Apple Silicon to ensure stability
            # But here let's use the model's auto-detection (which prefers CoreML on Mac)
            tts = Supertonic2TTS()
            tts.synthesize(
                text=text, output_path=output_path, voice_style=supertonic2_voice,
                lang_code=supertonic2_lang, speed=speed, steps=supertonic2_steps
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
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        gr.Info("Synthesis complete!")
        return output_path
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise gr.Error(f"Synthesis failed: {str(e)}")

def on_model_change(model_name: str):
    """Updates visibility of model-specific parameter groups."""
    cloning_models = ["Chatterbox", "Marvis", "CosyVoice", "NeuTTS Air"]
    ref_text_models = ["Marvis", "CosyVoice", "NeuTTS Air"]
    
    # Models that support MLX
    mlx_models = ["Kokoro", "Chatterbox", "CosyVoice"]
    # Models that support Streaming
    stream_models = ["Kokoro", "Kitten", "Supertone", "Marvis", "Chatterbox"]
    
    desc = MODEL_DESCRIPTIONS.get(model_name, "")
    
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
        gr.update(visible=(model_name == "Supertonic-2")),
        gr.update(visible=(model_name == "Dia2")),
        gr.update(visible=(model_name in mlx_models)), # use_mlx
        gr.update(visible=(model_name in stream_models)), # stream
        gr.update(value=desc, visible=bool(desc)) # description
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
                    choices=["Kokoro", "Kitten", "Chatterbox", "Marvis", "CosyVoice", "NeuTTS Air", "Supertone", "Supertonic-2", "Dia2"],
                    value="Kokoro",
                    interactive=True
                )
                
                model_desc = gr.Markdown(value=MODEL_DESCRIPTIONS["Kokoro"])

                text_input = gr.Textbox(label="Input Text", placeholder="Text to synthesize...", lines=4)
                
                ref_audio_input = gr.Audio(label="Reference Audio (Cloning)", type="filepath", visible=False)
                ref_text_input = gr.Textbox(label="Reference Text (Required if no .txt file exists next to audio)", visible=False)

                with gr.Accordion("⚙️ Global Settings", open=True):
                    with gr.Row():
                        speed = gr.Slider(label="Speed", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
                        use_mlx = gr.Checkbox(label="MLX Acceleration", value=True, visible=True)
                        stream = gr.Checkbox(label="Enable Streaming", value=False, visible=True)

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
                    with gr.Row():
                        kitten_version = gr.Dropdown(label="Version", choices=["0.1", "0.2"], value="0.2")
                        kitten_voice = gr.Dropdown(
                            label="Voice", 
                            choices=[
                                "expr-voice-2-m", "expr-voice-2-f", 
                                "expr-voice-3-m", "expr-voice-3-f", 
                                "expr-voice-4-m", "expr-voice-4-f", 
                                "expr-voice-5-m", "expr-voice-5-f"
                            ], 
                            value="expr-voice-4-f"
                        )

                with gr.Group(visible=False) as chatter_params:
                    gr.Markdown("### Chatterbox Settings")
                    with gr.Row():
                        chatter_exaggeration = gr.Slider(label="Exaggeration", minimum=0.0, maximum=1.0, value=0.7)
                        chatter_cfg = gr.Slider(label="CFG Weight", minimum=0.0, maximum=1.0, value=0.5)
                    with gr.Row():
                        chatter_lang = gr.Dropdown(label="Language", choices=CHATTERBOX_LANGS, value="en", allow_custom_value=True)
                        chatter_multi = gr.Checkbox(label="Multilingual Mode", value=False)
                    with gr.Row():
                        chatter_model_id = gr.Dropdown(label="Model ID", choices=CHATTERBOX_MODELS, value="mlx-community/chatterbox-turbo-4bit", allow_custom_value=True)
                        chatter_voice = gr.Dropdown(label="Voice Preset (Optional)", choices=CHATTERBOX_VOICES, allow_custom_value=True)

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
                    supertone_preset = gr.Dropdown(label="Voice Preset", choices=["F1", "F2", "M1", "M2"], value="F1")
                    with gr.Row():
                        supertone_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, value=8, step=1)
                        supertone_cfg = gr.Slider(label="CFG Scale", minimum=0.0, maximum=5.0, value=1.0)

                with gr.Group(visible=False) as supertonic2_params:
                    gr.Markdown("### Supertonic-2 Settings")
                    with gr.Row():
                        supertonic2_lang = gr.Dropdown(
                            label="Language", 
                            choices=[("English", "en"), ("Korean", "ko"), ("Spanish", "es"), ("Portuguese", "pt"), ("French", "fr")], 
                            value="en"
                        )
                        supertonic2_voice = gr.Dropdown(
                            label="Voice", 
                            choices=["F1", "F2", "M1", "M2"], 
                            value="F1"
                        )
                    supertonic2_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=30, value=10, step=1)

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
                supertone_params, supertonic2_params, dia2_params,
                use_mlx, stream, model_desc
            ]
        )
        
        
        kokoro_lang.change(
            fn=on_kokoro_lang_change,
            inputs=[kokoro_lang],
            outputs=[kokoro_voice]
        )
        
        chatter_lang.change(
            fn=on_chatter_lang_change,
            inputs=[chatter_lang],
            outputs=[chatter_voice]
        )

        generate_btn.click(
            fn=generate_speech,
            inputs=[
                model_dropdown, text_input, ref_audio_input, ref_text_input, speed, use_mlx, stream,
                kokoro_lang, kokoro_voice,
                kitten_version, kitten_voice,
                chatter_exaggeration, chatter_cfg, chatter_lang, chatter_multi, chatter_model_id, chatter_voice,
                marvis_temp, marvis_top_p, marvis_quant,
                cosy_instruct,
                neutts_backbone,
                supertone_preset, supertone_steps, supertone_cfg,
                supertonic2_lang, supertonic2_voice, supertonic2_steps,
                dia2_cfg, dia2_temp, dia2_top_k
            ],
            outputs=[audio_output]
        )
        
    return tts_layout

def on_chatter_lang_change(lang_code: str):
    """Filters Chatterbox voices based on language."""
    # Prefix mapping
    prefixes = {
        "en": ("af_", "bf_", "am_", "bm_", "en"),
        "es": ("ef_", "em_", "es"),
        "fr": ("ff_", "fr"),
        "it": ("if_", "im_", "it"),
        "pt": ("pf_", "pm_", "pt"),
        "de": ("de_", "df_", "dm_"),
        "ru": ("ru_", "rf_", "rm_"),
        "tr": ("tr_", "tf_", "tm_"),
        "hi": ("hi_", "hf_", "hm_"),
        "ja": ("jf_", "jm_", "ja"),
        "zh": ("zf_", "zm_", "zh"),
    }
    
    valid_prefixes = prefixes.get(lang_code, (lang_code,))
    
    filtered_voices = [
        v for v in CHATTERBOX_VOICES 
        if v.startswith(valid_prefixes) or v == lang_code
    ]
    
    # Always include the language code itself as a preset if it exists in VOICES
    if lang_code in CHATTERBOX_VOICES and lang_code not in filtered_voices:
        filtered_voices.insert(0, lang_code)
        
    if not filtered_voices:
        # Fallback to all voices if no match (or empty)
        filtered_voices = CHATTERBOX_VOICES
        
    return gr.update(choices=filtered_voices, value=filtered_voices[0] if filtered_voices else None)