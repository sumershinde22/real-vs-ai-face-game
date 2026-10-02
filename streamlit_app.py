import json
import os
import random

import numpy as np
import streamlit as st
from PIL import Image

try:
    import keras
except ImportError:
    from tensorflow import keras

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "test_images")
with open(os.path.join(HERE, "model_config.json")) as f:
    CFG = json.load(f)
ROUNDS = 10


@st.cache_resource
def get_model():
    return keras.saving.load_model(os.path.join(HERE, "best_model.keras"))


def model_score(fname):
    img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
    arr = np.asarray(img.resize((CFG["img_size"], CFG["img_size"])), dtype="float32") / 255.0
    return float(get_model().predict(arr[None, ...] * CFG["input_scale"], verbose=0).ravel()[0])


def is_real(fname):
    return fname.startswith("real")


def new_game():
    files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
    st.session_state.deck = random.sample(files, min(ROUNDS, len(files)))
    st.session_state.round = 0
    st.session_state.human = 0
    st.session_state.model = 0
    st.session_state.answer = None


def next_round():
    st.session_state.answer = None


def answer(guess_real):
    fname = st.session_state.deck[st.session_state.round]
    score = model_score(fname)
    truth = is_real(fname)
    human_ok = guess_real == truth
    model_ok = (score >= 0.5) == truth
    st.session_state.human += human_ok
    st.session_state.model += model_ok
    st.session_state.round += 1
    st.session_state.answer = (truth, score, human_ok, model_ok)


st.set_page_config(page_title="Real vs. AI Face Detector", page_icon="🕵️")
if "deck" not in st.session_state:
    new_game()

st.title("Can you spot the AI face?")
st.caption(f"You vs. a {CFG['model_name']} classifier "
           f"({100 * CFG['test_accuracy']:.1f}% on the held-out course test set).")

awaiting = st.session_state.answer is None
done = st.session_state.round >= len(st.session_state.deck)
idx = min(st.session_state.round if awaiting else st.session_state.round - 1,
          len(st.session_state.deck) - 1)

st.subheader(f"Round {idx + 1} of {ROUNDS}")
st.image(Image.open(os.path.join(IMG_DIR, st.session_state.deck[idx])), width=320)

if awaiting:
    col1, col2 = st.columns(2)
    col1.button("Real photo", use_container_width=True, on_click=answer, args=(True,))
    col2.button("AI-generated", use_container_width=True, on_click=answer, args=(False,))
else:
    truth, score, human_ok, model_ok = st.session_state.answer
    st.write(f"It was **{'a real photograph' if truth else 'AI-generated'}**.")
    (st.success if human_ok else st.error)(
        "You were right." if human_ok else "You got it wrong.")
    (st.success if model_ok else st.error)(
        f"The model said **{'Real' if score >= 0.5 else 'AI'}** "
        f"(P(real) = {score:.2f}) - {'correct' if model_ok else 'wrong'}.")
    if done:
        h, m = st.session_state.human, st.session_state.model
        st.header(f"You {h} - {m} Model" if h != m else f"Tie, {h} - {m}")
        st.write("You win!" if h > m else "The model wins." if m > h else "Dead heat.")
        st.button("Play again", on_click=new_game)
    else:
        st.button("Next image", on_click=next_round)

played = st.session_state.round
st.divider()
c1, c2 = st.columns(2)
c1.metric("You", f"{st.session_state.human} / {played}",
          f"{100 * st.session_state.human / played:.0f}%" if played else "-")
c2.metric("Model", f"{st.session_state.model} / {played}",
          f"{100 * st.session_state.model / played:.0f}%" if played else "-")
