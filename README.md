# Arabic to English Translation Model

This project is a neural machine translation model that translates Arabic text to English using a Seq2Seq architecture with an attention mechanism. The model has been built using PyTorch, and the application is deployed using Streamlit for easy interaction.

## Features
- Translates Arabic text to English in real-time.
- Uses a deep learning model built on Seq2Seq architecture with attention.
- Streamlit interface for easy and user-friendly access.
- Stores translation history.

## How to Run Locally
1. Clone the repository:
    ```bash
    git clone https://github.com/saadrehman171000/arabic_to_english.git
    cd arabic_to_english
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```

## Model Details
- **Encoder**: Bidirectional GRU.
- **Decoder**: GRU with attention.
- **Embedding Dimension**: 512.
- **Hidden Dimension**: 1024.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

