import gradio as gr
import logging
import json

logger = logging.getLogger(__name__)

def detect_speech_segments(
    audio_path: str,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_pad_ms: int
) -> str:
    """
    Detects speech segments using HumAware-VAD with custom parameters.
    """
    if not audio_path:
        raise gr.Error("Please upload an audio file to analyze.")
        
    try:
        from src.voice_cloning.vad.humaware import HumAwareVAD
        model = HumAwareVAD()
        segments = model.detect_speech(
            audio_path,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms
        )
        
        # Format as pretty JSON for display
        return json.dumps(segments, indent=2)
            
    except Exception as e:
        logger.error(f"VAD failed: {e}")
        raise gr.Error(f"VAD failed: {str(e)}")

def create_vad_tab():
    """Creates the VAD tab content."""
    with gr.Column() as vad_layout:
        gr.Markdown("## Voice Activity Detection")
        gr.Markdown("Identify speech segments in an audio file using HumAware-VAD.")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Upload Audio",
                    type="filepath"
                )
                
                with gr.Group():
                    gr.Markdown("### VAD Settings")
                    threshold = gr.Slider(
                        label="Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05
                    )
                    min_speech_ms = gr.Slider(
                        label="Min Speech Duration (ms)",
                        minimum=50,
                        maximum=2000,
                        value=250,
                        step=50
                    )
                    min_silence_ms = gr.Slider(
                        label="Min Silence Duration (ms)",
                        minimum=50,
                        maximum=2000,
                        value=100,
                        step=50
                    )
                    speech_pad_ms = gr.Slider(
                        label="Speech Padding (ms)",
                        minimum=0,
                        maximum=500,
                        value=30,
                        step=10
                    )

                analyze_btn = gr.Button("Analyze Speech", variant="primary")
            
            with gr.Column():
                segments_output = gr.Textbox(
                    label="Speech Segments (JSON)",
                    placeholder="Segments will appear here...",
                    lines=20
                )
        
        # Event wiring
        analyze_btn.click(
            fn=detect_speech_segments,
            inputs=[
                audio_input,
                threshold,
                min_speech_ms,
                min_silence_ms,
                speech_pad_ms
            ],
            outputs=[segments_output]
        )
        
    return vad_layout