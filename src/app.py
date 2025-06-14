import gradio as gr
import inference


def classify_tokens(image, annots_file, annots_format):
    if image is None:
        raise gr.Error("Please upload an image.")
    if annots_file is None:
        raise gr.Error("Please upload an annotation file.")
    if annots_format is None:
        raise gr.Error("Please choose annotation foramt.")

    annot_image, labeled_annot_file = inference.perform_inference(
        image, annots_file, annots_format
    )

    return annot_image, labeled_annot_file


iface = gr.Interface(
    fn=classify_tokens,
    inputs=[
        gr.Image(type="pil", label="Input image"),
        gr.File(label="Annotation file"),
        gr.Dropdown(
            choices=["STD", "ICPR22", "EconBiz & CHIMIE-R"],
            label="Annotation file format",
            value="STD",
        ),
    ],
    outputs=[
        gr.Image(type="pil", label="Annotated image"),
        gr.File(label="Labeled annotation file"),
    ],
    title="Chart Text Role Classification Demo",
    description="Upload an image and an annotation file to classify token roles. The application will return the annotated image and the labeled annotation file.",
)

if __name__ == "__main__":
    iface.launch()
