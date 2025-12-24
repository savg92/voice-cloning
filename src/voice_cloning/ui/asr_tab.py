import gradio as gr

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
        
    return asr_layout
