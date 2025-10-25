"""
🚢 Titanic Survival Prediction Web App with SHAP Explainability
"""

# --------------------------------------------------
# Imports
# --------------------------------------------------
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image
import shap
import streamlit.components.v1 as components

# --------------------------------------------------
# Streamlit Page Setup
# --------------------------------------------------
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")
st.title("🚢 Predict Your Survival Chances on the Titanic")

img = Image.open("titanic.jpg")
st.image(img, width=400)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
df = pd.read_csv("titanic.csv").drop(columns="Name")
df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

X = df.drop("Survived", axis=1).values.astype(np.float32)
y = df["Survived"].values.astype(np.float32)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

feature_names = ["Pclass", "Sex", "Age", "Siblings", "Children", "Fare"]

# --------------------------------------------------
# Helper to display SHAP plots in Streamlit
# --------------------------------------------------
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# --------------------------------------------------
# Load or Train Model
# --------------------------------------------------
@st.cache_resource
def load_or_train_model():
    if not os.path.exists("titanic.h5"):
        st.info("🧠 Training model, please wait...")

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(6,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])

        model.compile(optimizer="adam",
                      loss="binary_crossentropy",
                      metrics=["accuracy"])

        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=8,
            validation_data=(X_val, y_val),
            verbose=0
        )

        model.save("titanic.h5")
        with open("training_history", "wb") as f:
            pickle.dump(history.history, f)
        st.success("✅ Model training complete!")
        return model, history.history
    else:
        st.info("📂 Loading saved model...")
        model = load_model("titanic.h5")
        history = pickle.load(open("training_history", "rb"))
        return model, history

model, history = load_or_train_model()

# --------------------------------------------------
# Sidebar: User Inputs
# --------------------------------------------------
def get_user_input():
    st.sidebar.header("Passenger Information")

    Pclass = st.sidebar.slider("Passenger Class (1=First, 3=Third)", 1, 3, 2)
    sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
    age = st.sidebar.slider("Age", 0, 100, 30)
    n_siblings = st.sidebar.slider("# of Siblings", 0, 5, 0)
    n_children = st.sidebar.slider("# of Children", 0, 5, 0)
    fare = st.sidebar.slider("Ticket Price (£)", 0, 870, 50)

    sex = 1 if sex == "Male" else 0
    data = [[Pclass, sex, age, n_siblings, n_children, fare]]
    return tf.constant(data, dtype=tf.float32)

user_input = get_user_input()

# --------------------------------------------------
# Predict Survival
# --------------------------------------------------
prediction = model.predict(user_input, verbose=0)
survival_prob = float(prediction[0][0]) * 100
st.subheader("🎯 Prediction Result")
st.write(f"Your predicted chance of survival is **{survival_prob:.1f}%**")

# Progress Bar
progress_value = float(min(max(survival_prob / 100, 0), 1))
st.progress(progress_value)

# Survival Message
if survival_prob >= 75:
    st.success("🌟 Highly likely to survive!")
elif survival_prob >= 50:
    st.info("🙂 Fair chance of survival.")
elif survival_prob >= 25:
    st.warning("😟 Low chance of survival.")
else:
    st.error("💀 Unlikely to survive.")

# --------------------------------------------------
# Training Performance Plots
# --------------------------------------------------
def plot_training(history):
    acc_key = "accuracy" if "accuracy" in history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"

    epochs = range(1, len(history["loss"]) + 1)

    # Loss Plot
    st.subheader("📉 Training and Validation Loss")
    st.info("""
    **What this means:**
    - *Training Loss* shows how well the model is learning from the data it was trained on.
    - *Validation Loss* shows how well the model performs on unseen data (used to detect overfitting).
    - Ideally, both lines go down together.  
    - If validation loss starts increasing while training loss decreases, the model is overfitting.
    """)
    
    fig, ax = plt.subplots()
    plt.plot(epochs, history["loss"], "g", label="Training Loss")
    plt.plot(epochs, history["val_loss"], "b", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    st.pyplot(fig)

    # Accuracy Plot
    st.subheader("📈 Training and Validation Accuracy")
    st.info("""
    **Accuracy plot explanation:**  
    - **Training Accuracy (green):** Fraction of correct predictions on training data.  
    - **Validation Accuracy (blue):** Fraction of correct predictions on unseen data.  
    - Higher accuracy = better performance.  
    - Look for convergence between training and validation accuracy.
    """)
    
    fig, ax = plt.subplots()
    plt.plot(epochs, history[acc_key], "g", label="Training Accuracy")
    plt.plot(epochs, history[val_acc_key], "b", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    st.pyplot(fig)

plot_training(history)

# --------------------------------------------------
# SHAP Explainability
# --------------------------------------------------
def explain_prediction(model, input_data):
    st.subheader("🔍 Why the model made this prediction")
    st.write("""
    SHAP (SHapley Additive exPlanations) shows how each feature contributed to your survival prediction.
    Pink bars mean the feature increased your chance of survival; blue bars mean it decreased.
    """)

    # Convert Tensor to NumPy
    input_np = input_data.numpy()
    background = X_train[:100].astype(np.float32)  # ensure float32
    explainer = shap.DeepExplainer(model, background)

    # Get shap values and convert to NumPy
    shap_values = explainer.shap_values(input_np)
    sv = np.array(shap_values[0]).reshape(-1)  # flatten
    ev = float(explainer.expected_value[0])    # convert to float

    feature_names = ["Pclass", "Sex", "Age", "Siblings", "Children", "Fare"]

    # Force plot (works with Streamlit)
    force_plot = shap.force_plot(ev, sv, features=input_np[0].astype(np.float32), feature_names=feature_names)
    st_shap(force_plot, 400)

    # Bar chart
    contributions = sv
    st.subheader("📊 Feature Contributions")
    contribution_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contributions
    }).sort_values(by="Contribution", key=abs, ascending=False)

    colors = ["green" if x > 0 else "red" for x in contribution_df["Contribution"]]
    fig, ax = plt.subplots()
    ax.barh(contribution_df["Feature"], contribution_df["Contribution"], color=colors)
    ax.set_xlabel("Impact on Survival Probability")
    ax.set_ylabel("Feature")
    st.pyplot(fig)


explain_prediction(model, user_input)
