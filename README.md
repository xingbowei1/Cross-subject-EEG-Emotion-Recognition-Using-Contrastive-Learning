# Cross-subject EEG Emotion Recognition Using Contrastive Learning

## Overview
This project focuses on recognizing emotions across different subjects using EEG signals and contrastive learning techniques. The goal is to develop a model that effectively reduces individual differences in EEG to achieve accurate emotion recognition across various subjects.

## Features
- **Cross-subject Analysis**: Addresses variability in EEG data across different individuals.
- **Contrastive Learning**: Utilized to minimize individual differences and enhance model robustness and accuracy.
- **Customizable Architectures**: Multiple model configurations for experimentation.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/xingbowei1/Cross-subject-EEG-Emotion-Recognition-Using-Contrastive-Learning.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Prepare your dataset.
2. Configure your model and training parameters in `main.py`.
3. Run the training script:
    ```bash
    python main.py
    ```

## File Descriptions
- `data.py`: Data preprocessing scripts.
- `layers.py`: Custom neural network layers.
- `losses.py`: Loss function definitions.
- `model.py`: Main model architecture.
- `model_n.py`: Alternative model architecture.
- `util.py` and `utils.py`: Utility functions.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License
This project is licensed under the MIT License.
