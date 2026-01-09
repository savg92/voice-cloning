import gradio as gr
import tempfile
import logging
import torch
import os

logger = logging.getLogger(__name__)

# Kokoro voice mapping
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


MODEL_DESCRIPTIONS = {
    "Kokoro": "⚡ **Fast & High Quality** (82M params). Best overall for English/European languages. Supports streaming.",
    "Kitten": "🐱 **Lightweight** (30M params). Optimized for very low latency.",
    "Chatterbox": "💬 **Conversational** (460M params). Natural prosody and zero-shot cloning. Supports native [laugh], [cough] tags.",
    "Chatterbox-Turbo": "🚀 **Hybrid Engine**. High speed (EN) + Multilingual (23 Langs). Optimized for long-form content. Includes Resemble PerTh watermarking for safety.",
    "Marvis": "🤖 **Minimalist**. Simple English TTS.",
    "CosyVoice": "🎙️ **Advanced Generation**. Instruct-based control.",
    "NeuTTS Air": "💨 **CPU Optimized**. GGUF based models.",
    "Supertone": "🎛️ **Controllable**. Step-based generation with CFG (v1).",
    "Supertonic-2": "🎙️ **Fast & Multilingual**. New v2 model supporting EN, KO, ES, PT, FR via ONNX.",
    "Dia2": "🎹 **High Fidelity**. Diffusion-based (Requires CUDA).",
}

KOKORO_LANGS = {
    "a": "US English", "b": "British English", "e": "Spanish", "f": "French",
    "h": "Hindi", "i": "Italian", "j": "Japanese", "p": "Portuguese", "z": "Chinese"
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
    # Marvis
    marvis_temp: float,
    marvis_top_p: float,
    marvis_quant: bool,
    # CosyVoice
    cosy_instruct: str,
    # NeuTTS Air
    neutts_backbone: str,
    # Chatterbox
    chatterbox_lang: str,
    chatterbox_exaggeration: float,
    chatterbox_cfg: float,
    chatterbox_watermark: bool,
    # Supertone
    supertone_preset: str,
    supertone_steps: int,
    supertone_cfg: float,
    # Supertonic-2
    supertonic2_lang: str,
    supertonic2_voice: str,
    supertonic2_steps: int,
    supertonic2_pitch: float,
    supertonic2_energy: float,
    # Dia2
    dia2_cfg: float,
    dia2_temp: float,
    dia2_top_k: int
) -> str:
    if not text.strip():
        raise gr.Error("Please enter some text to synthesize.")
        
    cloning_models = ["Chatterbox", "Chatterbox-Turbo", "Marvis", "CosyVoice", "NeuTTS Air"]
    
    # Validation
    if model_name in ["Marvis", "CosyVoice", "NeuTTS Air"] and not reference_audio:
        raise gr.Error(f"Reference audio is required for model '{model_name}'.")

    output_path = tempfile.mktemp(suffix=".wav")
    
    try:
        gr.Info(f"Synthesizing with {model_name}...")
        
        if model_name == "Kokoro":
            from voice_cloning.tts.kokoro import synthesize_speech as kokoro_synthesize
            kokoro_synthesize(
                text=text, output_path=output_path, voice=kokoro_voice,
                lang_code=kokoro_lang, speed=speed, use_mlx=use_mlx, stream=stream
            )
            
        elif model_name == "Kitten":
            from voice_cloning.tts.kitten_nano import KittenNanoTTS
            v_map = {"0.1": "KittenML/kitten-tts-nano-0.1", "0.2": "KittenML/kitten-tts-nano-0.2"}
            model_id = v_map.get(kitten_version, "KittenML/kitten-tts-nano-0.2")
            tts = KittenNanoTTS(model_id=model_id)
            tts.synthesize_to_file(text=text, output_path=output_path, voice=kitten_voice, speed=speed, stream=stream)
            

        elif model_name == "Marvis":
            from voice_cloning.tts.marvis import MarvisTTS
            tts = MarvisTTS()
            tts.synthesize(
                text=text, output_path=output_path, ref_audio=reference_audio,
                ref_text=reference_text, speed=speed, temperature=marvis_temp,
                quantized=marvis_quant, stream=stream, lang_code="en"
            )
        
        elif model_name == "CosyVoice":
            from voice_cloning.tts.cosyvoice import synthesize_speech as cosy_synthesize
            cosy_synthesize(
                text=text, output_path=output_path, ref_audio_path=reference_audio,
                ref_text=reference_text, instruct_text=cosy_instruct,
                speed=speed, use_mlx=use_mlx
            )

        elif model_name == "NeuTTS Air":
            from voice_cloning.tts.neutts_air import synthesize_with_neutts_air
            synthesize_with_neutts_air(
                text=text, output_path=output_path, ref_audio=reference_audio,
                ref_text=reference_text, backbone=neutts_backbone, device="cpu"
            )

        elif model_name == "Supertone":
            from voice_cloning.tts.supertone import synthesize_with_supertone
            synthesize_with_supertone(
                text=text, output_path=output_path, preset=supertone_preset,
                steps=supertone_steps, cfg_scale=supertone_cfg, stream=stream
            )

        elif model_name == "Chatterbox":
            from voice_cloning.tts.chatterbox import synthesize_with_chatterbox
            synthesize_with_chatterbox(
                text=text, output_wav=output_path, source_wav=reference_audio,
                exaggeration=chatterbox_exaggeration, cfg_weight=chatterbox_cfg,
                language=chatterbox_lang, use_mlx=use_mlx, stream=stream
            )

        elif model_name == "Chatterbox-Turbo":
            from voice_cloning.tts.chatterbox_turbo import synthesize_with_chatterbox_turbo
            synthesize_with_chatterbox_turbo(
                text=text, output_wav=output_path, source_wav=reference_audio,
                exaggeration=chatterbox_exaggeration, cfg_weight=chatterbox_cfg,
                language=chatterbox_lang, use_mlx=use_mlx, stream=stream,
                watermark=chatterbox_watermark
            )

        elif model_name == "Supertonic-2":
            from voice_cloning.tts.supertonic2 import Supertonic2TTS
            tts = Supertonic2TTS()
            tts.synthesize(
                text=text, output_path=output_path, voice_style=supertonic2_voice,
                lang_code=supertonic2_lang, speed=speed, steps=supertonic2_steps,
                pitch_shift=supertonic2_pitch, energy_scale=supertonic2_energy
            )

        elif model_name == "Dia2":
            if not torch.cuda.is_available(): raise gr.Error("Requires CUDA.")
            from voice_cloning.tts.dia2 import Dia2TTS
            tts = Dia2TTS()
            tts.synthesize(text=text, output_path=output_path, cfg_scale=dia2_cfg, temperature=dia2_temp, top_k=dia2_top_k)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        gr.Info("Synthesis complete!")
        return output_path
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise gr.Error(f"Synthesis failed: {str(e)}")

def on_model_change(model_name: str):
    cloning_models = ["Chatterbox", "Chatterbox-Turbo", "Marvis", "CosyVoice", "NeuTTS Air"]
    ref_text_models = ["Marvis", "CosyVoice", "NeuTTS Air"]
    mlx_models = ["Kokoro", "CosyVoice", "Chatterbox", "Chatterbox-Turbo"]
    stream_models = ["Kokoro", "Kitten", "Supertone", "Marvis", "Chatterbox", "Chatterbox-Turbo"]
    
    desc = MODEL_DESCRIPTIONS.get(model_name, "")
    
    return [
        gr.update(visible=(model_name in cloning_models)), # ref_audio
        gr.update(visible=(model_name in ref_text_models)), # ref_text
        gr.update(visible=(model_name == "Kokoro")),
        gr.update(visible=(model_name == "Kitten")),
        gr.update(visible=(model_name in ["Chatterbox", "Chatterbox-Turbo"])), # chatterbox_params
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
    voices = KOKORO_VOICES.get(lang_code, ["af_heart"])
    return gr.update(choices=voices, value=voices[0])


def create_tts_tab():
    with gr.Column() as tts_layout:
        gr.Markdown("## 🎭 Text-to-Speech & Voice Cloning")
        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    label="Model Engine",
                    choices=["Kokoro", "Kitten", "Chatterbox", "Chatterbox-Turbo", "Marvis", "CosyVoice", "NeuTTS Air", "Supertone", "Supertonic-2", "Dia2"],
                    value="Kokoro", interactive=True
                )
                model_desc = gr.Markdown(value=MODEL_DESCRIPTIONS["Kokoro"])
                text_input = gr.Textbox(label="Input Text", placeholder="Text to synthesize...", lines=4)
                ref_audio_input = gr.Audio(label="Reference Audio (Cloning)", type="filepath", visible=False)
                ref_text_input = gr.Textbox(label="Reference Text", visible=False)

                with gr.Accordion("⚙️ Global Settings", open=True):
                    with gr.Row():
                        speed = gr.Slider(label="Speed", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
                        use_mlx = gr.Checkbox(label="MLX Acceleration", value=True)
                        stream = gr.Checkbox(label="Enable Streaming", value=False)

                # --- Model Groups ---
                with gr.Group(visible=True) as kokoro_params:
                    gr.Markdown("### Kokoro Settings")
                    with gr.Row():
                        kokoro_lang = gr.Dropdown(label="Language", choices=[(v, k) for k, v in KOKORO_LANGS.items()], value="a")
                        kokoro_voice = gr.Dropdown(label="Voice", choices=KOKORO_VOICES["a"], value="af_heart")

                with gr.Group(visible=False) as kitten_params:
                    gr.Markdown("### Kitten Settings")
                    with gr.Row():
                        kitten_version = gr.Dropdown(label="Version", choices=["0.1", "0.2"], value="0.2")
                        kitten_voice = gr.Dropdown(label="Voice", choices=["expr-voice-2-m", "expr-voice-2-f", "expr-voice-3-m", "expr-voice-3-f", "expr-voice-4-m", "expr-voice-4-f", "expr-voice-5-m", "expr-voice-5-f"], value="expr-voice-4-f")


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

                with gr.Group(visible=False) as chatterbox_params:
                    gr.Markdown("### Chatterbox Settings")
                    chatterbox_lang = gr.Dropdown(
                        label="Language",
                        choices=[
                            ("Arabic", "ar"), ("Chinese", "zh"), ("Danish", "da"), ("Dutch", "nl"), ("English", "en"),
                            ("Finnish", "fi"), ("French", "fr"), ("German", "de"), ("Greek", "el"), ("Hebrew", "he"),
                            ("Hindi", "hi"), ("Italian", "it"), ("Japanese", "ja"), ("Korean", "ko"), ("Malay", "ms"),
                            ("Norwegian", "no"), ("Polish", "pl"), ("Portuguese", "pt"), ("Russian", "ru"),
                            ("Spanish", "es"), ("Swahili", "sw"), ("Swedish", "sv"), ("Turkish", "tr")
                        ],
                        value="en"
                    )
                    with gr.Row():
                        chatterbox_exaggeration = gr.Slider(
                            label="Exaggeration", minimum=0.0, maximum=2.0, value=0.5, step=0.1,
                            info="Expressiveness. Higher (0.8+) makes speech dramatic. Push to 1.5+ for intense paralinguistic tags."
                        )
                        chatterbox_cfg = gr.Slider(
                            label="CFG Scale", minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                            info="Guidance weight. Lower (~0.3) for faster/stable pacing. Use 0.0 for cross-language cloning."
                        )
                    
                    with gr.Row():
                        chatterbox_watermark = gr.Checkbox(
                            label="Responsible AI Watermark", value=True,
                            info="Applies Resemble PerTh perceptual watermarking for audio attribution."
                        )
                    
                    with gr.Accordion("🎭 Paralinguistic Tags Reference", open=False):
                        gr.Markdown("""
                        Include these tags in your text to trigger sounds or emotions. 
                        
                        **Note:** Tag names vary between English (Turbo) and Multilingual engines.
                        
                        | **Type** | **English (Turbo) Tags** | **Multilingual Tags** |
                        | :--- | :--- | :--- |
                        | **Sounds** | `[laugh]`, `[cough]`, `[chuckle]`, `[clear throat]`, `[gasp]`, `[groan]`, `[shush]` | `[laughter]`, `[cough]`, `[giggle]`, `[clear_throat]`, `[gasp]`, `[groan]`, `[shhh]` |
                        | **Actions** | - | `[sip]`, `[chew]`, `[sneeze]`, `[snore]`, `[inhale]`, `[exhale]`, `[humming]` |
                        | **Emotions** | `[happy]`, `[angry]`, `[crying]`, `[sarcastic]`, `[surprised]`, `[whispering]` | `[whisper]`, `[cry]`, `[guffaw]` |
                        | **Animals** | - | `[bark]`, `[meow]`, `[howl]` |
                        
                        *Example (EN): "[laugh] That is so funny! [whispering] But don't tell anyone."*
                        *Example (MTL): "[laughter] That is so funny! [whisper] But don't tell anyone."*

                        > [!TIP]
                        > **For Stronger Effects:** If tags feel too subtle, increase **Exaggeration** and **CFG Scale** to **1.0+**. Adding a space or period before/after a tag can also help the model transition more clearly.
                        """)

                with gr.Group(visible=False) as supertone_params:
                    gr.Markdown("### Supertone Settings")
                    supertone_preset = gr.Dropdown(label="Voice Preset", choices=["F1", "F2", "M1", "M2"], value="F1")
                    with gr.Row():
                        supertone_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, value=8, step=1)
                        supertone_cfg = gr.Slider(label="CFG Scale", minimum=0.0, maximum=5.0, value=1.0)

                with gr.Group(visible=False) as supertonic2_params:
                    gr.Markdown("### Supertonic-2 Settings")
                    with gr.Row():
                        supertonic2_lang = gr.Dropdown(label="Language", choices=[("English", "en"), ("Korean", "ko"), ("Spanish", "es"), ("Portuguese", "pt"), ("French", "fr")], value="en")
                        supertonic2_voice = gr.Dropdown(label="Voice", choices=["F1", "F2", "M1", "M2"], value="F1")
                    supertonic2_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=30, value=10, step=1)
                    with gr.Accordion("🎛️ Prosody Controls", open=False):
                        with gr.Row():
                            supertonic2_pitch = gr.Slider(label="Pitch Shift (semitones)", minimum=-12, maximum=12, value=0, step=0.5)
                            supertonic2_energy = gr.Slider(label="Energy/Emphasis", minimum=0.5, maximum=2.0, value=1.0, step=0.1)

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
        model_dropdown.change(fn=on_model_change, inputs=[model_dropdown], outputs=[ref_audio_input, ref_text_input, kokoro_params, kitten_params, chatterbox_params, marvis_params, cosy_params, neutts_params, supertone_params, supertonic2_params, dia2_params, use_mlx, stream, model_desc])
        kokoro_lang.change(fn=on_kokoro_lang_change, inputs=[kokoro_lang], outputs=[kokoro_voice])

        generate_btn.click(
            fn=generate_speech,
            inputs=[model_dropdown, text_input, ref_audio_input, ref_text_input, speed, use_mlx, stream, kokoro_lang, kokoro_voice, kitten_version, kitten_voice, marvis_temp, marvis_top_p, marvis_quant, cosy_instruct, neutts_backbone, chatterbox_lang, chatterbox_exaggeration, chatterbox_cfg, chatterbox_watermark, supertone_preset, supertone_steps, supertone_cfg, supertonic2_lang, supertonic2_voice, supertonic2_steps, supertonic2_pitch, supertonic2_energy, dia2_cfg, dia2_temp, dia2_top_k],
            outputs=[audio_output]
        )
    return tts_layout