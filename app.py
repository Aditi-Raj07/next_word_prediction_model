
import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LSTM Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)


# =========================================================
# LOAD MODEL + TOKENIZER + MAX LENGTH
# =========================================================

@st.cache_resource
def load_artifacts():

    # Load tokenizer
    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    # Load max length
    with open("max_len.pkl", "rb") as file:
        max_len = pickle.load(file)

    # Load trained LSTM model
    model = load_model("lstm_model.h5")

    return tokenizer, max_len, model


tokenizer, max_len, model = load_artifacts()


# =========================================================
# PREDICT NEXT WORD
# =========================================================

def predict_next_word(text):

    # Convert text into integer sequence
    token_list = tokenizer.texts_to_sequences([text])[0]

    if not token_list:
        return None

    # Use the most recent tokens
    token_list = token_list[-(max_len - 1):]

    # Pad input sequence
    token_list = pad_sequences(
        [token_list],
        maxlen=max_len - 1,
        padding="pre"
    )

    # Model prediction
    predictions = model.predict(
        token_list,
        verbose=0
    )

    # Get highest probability token
    predicted_index = np.argmax(
        predictions[0]
    )

    # Convert token ID -> word
    for word, index in tokenizer.word_index.items():

        if index == predicted_index:
            return word

    return None


# =========================================================
# HEADER
# =========================================================

st.title("🧠 LSTM Next Word Predictor")

st.write(
    "Enter a sentence and let the trained LSTM model "
    "predict the next word."
)

st.divider()


# =========================================================
# INPUT
# =========================================================

text = st.text_area(
    "✍️ Enter your text",
    placeholder="Example: I am going to",
    height=120
)


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🔮 Predict Next Word",
    use_container_width=True
):

    if not text.strip():

        st.warning(
            "⚠️ Please enter some text first."
        )

    else:

        next_word = predict_next_word(text)

        if next_word:

            st.success(
                "Prediction completed!"
            )

            st.subheader(
                "🎯 Predicted Next Word"
            )

            st.markdown(
                f"""
                <div style="
                    padding: 25px;
                    border-radius: 15px;
                    background-color: #f0f2f6;
                    text-align: center;
                    font-size: 36px;
                    font-weight: bold;
                    margin-top: 10px;
                    margin-bottom: 20px;
                ">
                    {next_word}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Show complete sentence
            st.write(
                f"**Your text:** {text}"
            )

            st.write(
                f"**Prediction:** {text} {next_word}"
            )

        else:

            st.error(
                "❌ Unable to predict the next word."
            )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.subheader("📊 Model Information")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Vocabulary Size",
        len(tokenizer.word_index) + 1
    )

with col2:

    st.metric(
        "Max Sequence Length",
        max_len
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🧠 About")

    st.write(
        """
        This application uses a trained
        **LSTM (Long Short-Term Memory)**
        neural network for next-word prediction.
        """
    )

    st.write("### Model")

    st.code(
        "lstm_model.h5"
    )

    st.write("### Tokenizer")

    st.code(
        "tokenizer.pkl"
    )

    st.write("### Max Length")

    st.code(
        "max_len.pkl"
    )

    st.divider()

    st.caption(
        "TensorFlow + Keras + Streamlit"
    )

