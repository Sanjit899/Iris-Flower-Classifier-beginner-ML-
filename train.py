import sys
import joblib
import numpy as np
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = "iris_model.pkl"

# --- Train and save model ---
def train_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODEL_FILE)
    print("✅ Model trained and saved as iris_model.pkl")

# --- Predict from CLI ---
def predict_cli(features):
    clf = joblib.load(MODEL_FILE)
    prediction = clf.predict([features])
    iris = load_iris()
    print(f"🌸 Predicted species: {iris.target_names[prediction][0]}")

# --- Streamlit Web App ---
def run_streamlit():
    st.title("🌸 Iris Flower Classifier")
    st.write("Enter the flower measurements to predict the species:")

    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
    petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

    if st.button("Predict"):
        clf = joblib.load(MODEL_FILE)
        features = [sepal_length, sepal_width, petal_length, petal_width]
        prediction = clf.predict([features])
        iris = load_iris()
        st.success(f"🌸 Predicted species: {iris.target_names[prediction][0]}")

# --- Entry Point ---
if __name__ == "__main__":
    # Streamlit sets this environment variable when running
    if st.runtime.exists():
        run_streamlit()
    else:
        if len(sys.argv) < 2:
            print(
                "Usage:\n"
                "  python train.py train                      -> Train and save model\n"
                "  python train.py predict <4 features>       -> Predict species\n"
                "     Example: python train.py predict 5.1 3.5 1.4 0.2\n"
                "  streamlit run train.py                     -> Run web app"
            )
        elif sys.argv[1] == "train":
            train_model()
        elif sys.argv[1] == "predict" and len(sys.argv) == 6:
            features = list(map(float, sys.argv[2:]))
            predict_cli(features)
        else:
            print("❌ Invalid command. Run without args for usage help.")
