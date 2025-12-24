import gradio as gr

def create_tts_tab():
    """Creates the TTS tab content."""
    with gr.Column() as tts_layout:
        gr.Markdown("## Text-to-Speech")
        
        with gr.Row():
            with gr.Column():
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=["Kokoro", "Kitten"],
                    value="Kokoro",
                    interactive=True
                )
                text_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text to synthesize...",
                    lines=3
                )
                generate_btn = gr.Button("Generate Speech", variant="primary")
            
            with gr.Column():
                audio_output = gr.Audio(label="Generated Audio", type="filepath")
        
        # Placeholder for event handling (to be implemented in next tasks)
        
    return tts_layout
