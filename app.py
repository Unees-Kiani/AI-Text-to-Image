import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient()

def generate_image(prompt):
    if prompt == "":
        return None

    image = client.text_to_image(
        prompt,
        model="black-forest-labs/FLUX.1-schnell"
    )

    return image


app = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(
        label="Enter your prompt",
        placeholder="A cute robot in a garden"
    ),
    outputs=gr.Image(label="Generated Image"),
    title="AI Text-to-Image Generator"
)

app.launch()