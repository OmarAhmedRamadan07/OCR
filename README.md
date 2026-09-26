# OCR — TrOCR (Fully Local, No API Key)

A fully local Optical Character Recognition (OCR) pipeline built with Microsoft's TrOCR model. It extracts text from handwritten or printed images without relying on any external API or internet connection at inference time. Built as part of the AI & Data Science Internship at ACUD (Administrative Capital for Urban Development).

## Overview

This project takes an uploaded image, preprocesses it, segments it into individual text lines, and runs each line through a TrOCR (Transformer-based OCR) model to extract text. An optional offline spell-checking step can clean up minor recognition errors in English text.

**Pipeline flow:**
1. Upload an image
2. Preprocessing (grayscale, contrast/sharpness enhancement, binarization)
3. Line segmentation using a horizontal projection profile
4. Text extraction per line using TrOCR
5. (Optional) Offline English spell-check

## Features

- Fully local inference — no API key, no internet required after the model is downloaded
- Works with both handwritten and printed text (model can be swapped for `microsoft/trocr-base-printed`)
- Automatic line segmentation for multi-line documents
- Image preprocessing pipeline (contrast, sharpness, Otsu thresholding)
- Optional manual rotation correction (0°, 90°, 180°, -90°)
- Optional offline spell-checking (English only) using `pyspellchecker`
- Interactive Streamlit web interface
- Downloadable extracted text as a `.txt` file
- Jupyter notebook version with an optional fine-tuning section for custom handwriting datasets

## Tech Stack

- **Model:** TrOCR (`microsoft/trocr-base-handwritten`) via Hugging Face Transformers
- **Deep Learning:** PyTorch
- **Image Processing:** OpenCV, Pillow
- **Web App:** Streamlit
- **Spell-Checking:** pyspellchecker

## Project Structure

```
OCR/
├── app.py               # Streamlit web application
├── OCR_Merged.ipynb     # Jupyter notebook (pipeline walkthrough + optional fine-tuning)
├── requirements.txt     # Python dependencies
└── README.md
```

## Installation

You have two options: run the app yourself from the terminal, or just open it directly from the live link with no setup at all.

```bash
git clone https://github.com/OmarAhmedRamadan07/OCR.git
cd OCR
pip install -r requirements.txt
```

## Usage

### Option 1 — Run the web app from the terminal

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, upload an image, and click **Run OCR**.

### Option 2 — Open it directly from the link (no source code needed)

You don't have to clone the repo or install anything at all. The app is already deployed and ready to use straight from your browser:

https://ocr-text-extractor.streamlit.app/

### Run the notebook

Open `OCR_Merged.ipynb`, set `IMAGE_PATH` to the image you want to process, and run the cells in order.

## How It Works

1. **Preprocessing** — the image is converted to grayscale, contrast and sharpness are enhanced, and Otsu's thresholding is applied to produce a clean binary image.
2. **Line Segmentation** — a horizontal projection profile of pixel intensities is used to detect gaps between lines of text and split the image into individual line crops.
3. **Text Extraction** — each line crop is passed to the TrOCR model, which generates the recognized text using its vision encoder–decoder architecture.
4. **Spell-Check (optional)** — each word in the extracted text is checked against an offline English dictionary and corrected if a close match is found.

## Fine-Tuning (Optional)

The notebook includes an optional section for fine-tuning TrOCR on a custom handwriting dataset (a CSV file with `image` and `text` columns), useful for improving accuracy on domain-specific or personal handwriting styles.

## Notes

- The first run downloads the TrOCR model weights from Hugging Face; subsequent runs use the cached model.
- Spell-checking currently supports English only.
- Best results are achieved with clear, well-lit, reasonably horizontal images. Use the rotation setting for scanned pages that are sideways or upside down.

## Acknowledgments

Built as part of the AI & Data Science Internship at ACUD (Administrative Capital for Urban Development), CET191 — Internship I, El Sewedy University of Technology.

## License

This project is provided for educational and portfolio purposes.
