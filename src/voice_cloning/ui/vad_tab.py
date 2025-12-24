import gradio as gr
import logging
import json

logger = logging.getLogger(__name__)

def detect_speech_segments(audio_path: str) -> str:
    """
    Detects speech segments using HumAware-VAD.
    
    Args:
        audio_path: The path to the audio file.
        
    Returns:
        A JSON string representing the speech segments.
    """
    if not audio_path:
        raise gr.Error("Please upload an audio file to analyze.")
        
    try:
        from src.voice_cloning.vad.humaware import HumAwareVAD
        model = HumAwareVAD()
        segments = model.detect_speech(audio_path)
        
        # Format as pretty JSON for display
        return json.dumps(segments, indent=2)
            
    except Exception as e:
        logger.error(f"VAD failed: {e}")
        raise gr.Error(f"VAD failed: {str(e)}")

def create_vad_tab():
    """Creates the VAD tab content."""
    with gr.Column() as vad_layout:
        gr.Markdown("## Voice Activity Detection")
        gr.Markdown("Identify speech segments in an audio file using HumAware-VAD (Silero fine-tune).")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Upload Audio",
                    type="filepath"
                )
                analyze_btn = gr.Button("Analyze Speech", variant="primary")
            
            with gr.Column():
                segments_output = gr.Textbox(
                    label="Speech Segments (JSON)",
                    placeholder="Segments will appear here...",
                    lines=15
                )
        
        # Event wiring
        analyze_btn.click(
            fn=detect_speech_segments,
            inputs=[audio_input],
            outputs=[segments_output]
        )
        
    return vad_layout
