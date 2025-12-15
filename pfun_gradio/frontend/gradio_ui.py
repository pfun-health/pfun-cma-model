"""Demo UI endpoint for LLM-based CMA parameter suggestions.

Uses Gradio for the interface. Hits the /llm/generate-scenario endpoint
to generate a scenario based on user input.
"""
import gradio as gr


def setup_gradio_ui(server_name: str = "0.0.0.0", server_port: int = 7860, **kwargs):
    """Set up the Gradio demo interface."""

    with gr.Blocks() as demo:
        gr.Markdown("# CMA Parameter Suggestion Demo")
        gr.Markdown(
            "This demo uses a Large Language Model (LLM) to suggest CMA model parameters "
            "based on a brief description of the user's condition. "
            "Enter a description below and click 'Generate Parameters' to see the suggestions."
        )
        description_input = gr.Textbox(
            label="Scenario Description (third-person)",
            placeholder="E.g., 'The patient has type 1 diabetes and struggle with high blood sugar after meals.'",
            lines=4
        )
        generate_button = gr.Button("Generate Parameters")
        output_box = gr.Textbox(
            label="Likely CMA Parameters",
            placeholder="CMA parameters will appear here...",
            lines=10
        )
        # @TODO: time series plotting of /model/run results could be added here later
        # ! ! ! !

        def generate_parameters(description):
            import requests
            response = requests.post(
                "/llm/generate-scenario",
                json={"description": description},
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("suggested_parameters", "No parameters returned.")
            else:
                return f"Error: {response.status_code} - {response.text}"

        generate_button.click(  # type: ignore
            fn=generate_parameters,
            inputs=[description_input],
            outputs=[output_box]
        )

    return demo


def launch_demo(server_name: str = "0.0.0.0", server_port: int = 7860, **kwargs):
    demo = setup_gradio_ui(server_name=server_name, server_port=server_port, **kwargs)
    demo.queue()
    return demo.launch(server_name=server_name, server_port=server_port, **kwargs)


if __name__ == "__main__":
    launch_demo()
