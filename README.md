# X-ray Image Classification with CNN

This project demonstrates how to use a Convolutional Neural Network (CNN) to classify X-ray images into categories, such as detecting covid or normal conditions. The project is implemented in Python using TensorFlow and Keras.

---

## Features

- **Model Training**: Trains a CNN to classify X-ray images.
- **Prediction**: Allows predictions by processing images programmatically.
- **Data Preprocessing**: Supports resizing, normalization, and augmentation of images.
- **Metrics**: Provides evaluation metrics such as accuracy on test data.

---

## File Structure

```
project/
├── xray_cnn_model.h5       # Pre-trained CNN model
├── xray_cnn.ipynb          # Jupyter notebook for model training
├── requirements.txt        # List of Python dependencies
└── README.md               # Project documentation
```

---

## Requirements

- Python 3.12.1 or later
- TensorFlow 2.0 or later
- Pillow
- NumPy

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## Usage Instructions

### 1. Train the Model
To train the CNN model, run the Jupyter Notebook `xray_cnn.ipynb` using any compatible editor such as JupyterLab or VS Code.

### 2. Predict Programmatically
Use the provided `predict_xray` function in `xray_cnn.ipynb` for programmatic predictions.

---

## How It Works

1. The dataset is preprocessed by resizing images, normalizing pixel values, and augmenting data to improve model generalization.
2. A CNN is trained on the dataset to classify X-ray images into predefined categories.
3. After training, the model is saved as `xray_cnn_model_t300.h5` and can be used for predictions on new images.

---

## Applications

- **Medical Diagnostics**: Automated detection of covid and other conditions in X-ray images.
- **Research**: Assisting in developing AI-based diagnostic tools.
- **Education**: Teaching students and professionals about deep learning applications in medical imaging.

---

## Future Enhancements

- **Web Interface**: Develop a user-friendly interface for non-programmers.
- **Additional Classes**: Extend the model to detect other diseases.
- **Explainability**: Integrate tools like Grad-CAM to visualize model decision-making.
- **Optimized Deployment**: Convert the model to TensorFlow Lite for mobile or edge device deployment.

---

## Dataset

The dataset should include X-ray images organized into directories based on class labels, for example:
```
path/to/dataset/
├── covid/
├── normal/
```
Images should be resized to match the model's input dimensions (e.g., 150x150 pixels).

---

## Evaluation
The trained model can be evaluated on a test dataset by using the code provided in the notebook. Key metrics include accuracy and loss.

---

## License

This project is licensed under the CC0-1.0 license License. Feel free to use and modify the code for personal or commercial purposes.

For more details, refer to the full license text at: [https://creativecommons.org/publicdomain/zero/1.0/](https://creativecommons.org/publicdomain/zero/1.0/)

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests with enhancements or fixes.

---

## Acknowledgments

- **TensorFlow**: For providing a robust framework for building and training deep learning models.
- **Open-Source Contributors**: For sharing datasets and code examples that made this project possible.
- **Educational Resources**: Tutorials and documentation that supported the development of this project.

---
