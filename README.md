# chart-text-role-classification-demo

This project is a Gradio application designed for token role classification inference. It allows users to upload an image and an annotation file, processes them, and returns an annotated image along with an updated annotation file.

## Project Structure

```
token-role-classification-app
├── src
│   ├── app.py            # Main entry point for the Gradio application
│   ├── inference.py      # Logic for token role classification inference
│   ├── utils.py          # Utility functions for image processing and annotations
│   └── types
│       └── __init__.py   # Custom types and data structures
├── requirements.txt       # Project dependencies
├── README.md              # Documentation for the project
└── .gitignore             # Files and directories to ignore by Git
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd token-role-classification-app
pip install -r requirements.txt
```

## Usage

To run the Gradio application, execute the following command:

```bash
python src/app.py
```

Once the application is running, you can access it in your web browser. Upload an image and an annotation file to perform token role classification.

## Functionality

- **Image Upload**: Users can upload an image for processing.
- **Annotation File Upload**: Users can upload an annotation file that corresponds to the image.
- **Inference**: The application processes the inputs and performs token role classification.
- **Output**: The application returns an annotated image and an updated annotation file, which can be downloaded by the user.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
