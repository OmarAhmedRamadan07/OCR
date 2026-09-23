"""
OCR App: TrOCR (fully local, no API key needed)

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import cv2
import torch
import streamlit as st

from PIL import Image, ImageEnhance
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    RobertaTokenizer,
    ViTImageProcessor,
)


# =========================================================
# CONFIG
# =========================================================

TROCR_MODEL_NAME = "microsoft/trocr-base-handwritten"


# =========================================================
# CACHED RESOURCES
# =========================================================

@st.cache_resource(show_spinner="Loading TrOCR model...")
def load_trocr():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Normal path.
        processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
    except ValueError:
        try:
            # Some transformers versions still fail to auto-build the fast
            # tokenizer here even with use_fast=False, so build the tokenizer
            # and image processor separately and assemble them by hand —
            # this skips the AutoProcessor code path entirely.
            tokenizer = RobertaTokenizer.from_pretrained(TROCR_MODEL_NAME)
            image_processor = ViTImageProcessor.from_pretrained(TROCR_MODEL_NAME)
            processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        except Exception as e2:
            raise RuntimeError(
                "Could not load the TrOCR processor/tokenizer. Try: "
                "pip install -U transformers tokenizers sentencepiece protobuf"
            ) from e2

    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
    model.to(device)
    return processor, model, device


@st.cache_resource(show_spinner=False)
def load_spellchecker():
    from spellchecker import SpellChecker
    return SpellChecker()


# =========================================================
# IMAGE PROCESSING
# =========================================================

def preprocess_image(image, rotate_fix=0):
    if rotate_fix:
        image = image.rotate(rotate_fix, expand=True)

    gray = image.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(1.5)
    gray = ImageEnhance.Sharpness(gray).enhance(1.5)

    img_cv = np.array(gray)
    _, thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(thresh).convert("RGB")


def segment_lines(image, min_gap=10, min_height=10):
    img = np.array(image.convert("L"))
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    horizontal_sum = np.sum(binary, axis=1)

    lines = []
    in_line = False
    start = 0

    for i, value in enumerate(horizontal_sum):
        if value > min_gap and not in_line:
            start = i
            in_line = True
        elif value <= min_gap and in_line:
            end = i
            if end - start > min_height:
                lines.append(image.crop((0, start, image.width, end)))
            in_line = False

    if not lines:
        lines = [image]

    return lines


def extract_text_from_line(processor, model, device, line_image):
    pixel_values = processor(images=line_image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_length=128)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def run_trocr(processor, model, device, image, rotate_fix=0):
    clean = preprocess_image(image, rotate_fix=rotate_fix)
    lines = segment_lines(clean)
    line_texts = [extract_text_from_line(processor, model, device, line) for line in lines]
    raw_text = "\n".join(line_texts)
    return raw_text, lines


def spell_correct(text, spell):
    """Very simple offline word-level spell correction (English only, no API needed)."""
    corrected_lines = []
    changes = []

    for line in text.split("\n"):
        words = line.split(" ")
        new_words = []
        for w in words:
            stripped = w.strip(".,!?;:\"'()")
            if stripped and stripped.isalpha() and stripped.lower() not in spell:
                suggestion = spell.correction(stripped.lower())
                if suggestion and suggestion != stripped.lower():
                    new_word = w.replace(stripped, suggestion)
                    changes.append(f"{stripped} → {suggestion}")
                    new_words.append(new_word)
                    continue
            new_words.append(w)
        corrected_lines.append(" ".join(new_words))

    return "\n".join(corrected_lines), changes


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="OCR: TrOCR", page_icon="📝", layout="wide")

st.title("📝 OCR — TrOCR (fully local, no API key)")
st.caption("Upload an image (handwritten or printed) and TrOCR extracts the text. Everything runs locally.")

with st.sidebar:
    st.header("Settings")

    rotate_fix = st.selectbox(
        "Rotate image before processing",
        options=[0, 90, 180, -90],
        format_func=lambda x: f"{x}°",
    )

    use_spellcheck = st.checkbox(
        "Enable offline spell-check (English only)",
        value=False,
        help="Simple local word correction using pyspellchecker. No internet or API key needed.",
    )

    st.divider()
    st.caption("Model: microsoft/trocr-base-handwritten")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Image")
        st.image(image, use_container_width=True)

    run_button = st.button("Run OCR", type="primary")

    if run_button:
        processor, model, device = load_trocr()

        with st.spinner("Extracting text from the image..."):
            raw_text, lines = run_trocr(processor, model, device, image, rotate_fix=rotate_fix)

        with col2:
            st.subheader("Extracted text")
            display_text = raw_text
            changes = []

            if use_spellcheck:
                with st.spinner("Running offline spell-check..."):
                    spell = load_spellchecker()
                    display_text, changes = spell_correct(raw_text, spell)

            st.text_area("text", display_text, height=250, label_visibility="collapsed")

            st.download_button(
                "⬇️ Download text",
                data=display_text,
                file_name="ocr_text.txt",
                mime="text/plain",
            )

            if use_spellcheck and changes:
                with st.expander(f"Spelling corrections made ({len(changes)})"):
                    for c in changes:
                        st.markdown(f"- {c}")

        with st.expander(f"Segmented lines ({len(lines)} lines)"):
            for i, line in enumerate(lines, start=1):
                st.image(line, caption=f"Line {i}")
else:
    st.info("Upload an image to get started.")
