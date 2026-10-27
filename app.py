import streamlit as st
from pathlib import Path

from poetry_gen.dataset import Dataset
from poetry_gen.model import load_model
from poetry_gen.generator import generate_text

# ---------------- Page configuration ----------------
st.set_page_config(
    page_title="AI Poetry Generator",
    page_icon="✍️",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "textgenerator.keras"
SEQ_LENGTH = 40

# ---------------- Load resources ----------------
@st.cache_resource
def load_resources():
    dataset = Dataset()
    dataset.load_data()
    model = load_model(str(MODEL_PATH))
    return dataset, model

# ---------------- User interface ----------------
st.title("✍️ AI Poetry Generator")
st.markdown(
    "Generate Shakespeare-inspired text using your trained "
    "LSTM character-level model."
)


if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH}\n\n"
        "Place textgenerator.keras in the same folder as app.py."
    )
    st.stop()

with st.sidebar:
    st.header("Generation Settings")

    length = st.slider(
        "Output length (characters)",
        min_value=100,
        max_value=1000,
        value=300,
        step=50
    )

    temperature = st.slider(
        "Creativity (temperature)",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Lower values are more predictable; higher values are more varied."
    )

    st.caption("The model uses a 40-character input sequence.")

seed = st.text_area(
    "Starting text (optional)",
    placeholder="Enter a phrase to begin the generation...",
    help="If provided, the model uses the ending of your phrase as its starting context."
)

generate_button = st.button(
    "✨ Generate Text",
    type="primary",
    use_container_width=True
)

if generate_button:
    try:
        with st.spinner("Loading model and generating text..."):
            dataset, model = load_resources()

            generated_text = generate_text(
                model=model,
                dataset=dataset,
                length=length,
                temperature=temperature,
                seed=seed.strip() or None,
                seq_length=SEQ_LENGTH
            )

        st.subheader("Generated Text")
        st.text_area(
            "Your generated text",
            value=generated_text,
            height=300,
            key="generated_output"
        )

        st.caption(
            f"Generated {len(generated_text)} characters, "
            f"including the initial seed. Temperature: {temperature:.1f}"
        )

        st.download_button(
            label="📥 Download Text",
            data=generated_text,
            file_name="generated_poetry.txt",
            mime="text/plain",
            use_container_width=True
        )

    except Exception as error:
        st.error(f"Text generation failed: {error}")
        st.info(
            "Check that the model file and project dependencies are available."
        )

st.divider()

