import gradio as gr


def greet(name: str, intensity: int) -> str:
    return "Hello, " + name + "!" * int(intensity)


demo = gr.Interface(
    fn=greet,
    inputs=[
        "text",
        "slider",
    ],  # the inputs are a text box and a slider ("text" and "slider" are components in Gradio)
    outputs=["text"],  # the output is a text box
)

demo.launch()
