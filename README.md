# Hand Gesture Drawing with Hybrid AI OCR Engine 🖐️🎨

A state-of-the-art Python application that combines real-time hand gesture tracking with a multi-tier Transformer-based OCR pipeline.

## 🚀 Key Features

### 1. Hand Gesture Control (MediaPipe)
- **Zero-Lag Drawing**: Uses the index finger for continuous virtual sketching.
- **Gesture Shortcuts**: 
  - 🖐️ **Open Palm**: Erase.
  - 🤏 **Finger Snap**: Toggle colors.
  - ✊ **Closed Fist**: Break current line.
- **Lighting Robustness**: Integrated **CLAHE** (Contrast Limited Adaptive Histogram Equalization) preprocessing to ensure tracking stability in harsh lighting.

### 2. Hybrid AI OCR Engine
When the session ends, the application analyzes the canvas using a sophisticated pipeline:
- **Detection (CRAFT)**: Locates text bounding boxes on the canvas.
- **Recognition (TrOCR)**: Uses **Microsoft's Vision Transformer (TrOCR)** for high-accuracy handwriting recognition.
- **Semantic Correction (DistilBERT)**: Triple-verifies results using **BPE tokenization** to ensure context-aware accuracy (e.g., correcting "H3LLO" to "HELLO").

## 🛠️ Tech Stack
- **Vision**: MediaPipe, OpenCV, EasyOCR, PIL
- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **NLP**: DistilBERT, TextBlob, Pyspellchecker
- **Visualization**: Matplotlib

## 📦 Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage
Run the main script:
```bash
python "object tracking.py"
```

### Controls:
- `c`: Clear canvas
- `q`: Quit and start OCR analysis

## 📄 License
MIT
