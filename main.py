import streamlit as st
import pandas as pd
import pickle
from utils import preprocess_transaction, get_groq_explanation
import os
from openai import OpenAI

st.set_page_config(page_title="Fraud Detection System", layout="wide")

# Initialize the OpenAI client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get('GROQ_API_KEY')
)

@st.cache_resource
def load_model():
    with open('fraud_detection_model_balanced.pkl', 'rb') as f:
        return pickle.load(f)

def analyze_transaction(transaction, model_package):
    """Analyze a single transaction and return prediction and explanation."""
    processed_transaction = preprocess_transaction(transaction, model_package['model'], 'XGBoost')
    probability = model_package['model'].predict_proba(processed_transaction)[:, 1][0]

    # Generate explanation using Groq AI
    explanation = get_groq_explanation(
        probability=probability,
        transaction_data=transaction,
        model_name='XGBoost',
        client=client
    )

    return probability, explanation

def main():
    st.title("🔍 Fraud Detection System")

    try:
        model_package = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    st.subheader("Upload or Select Transaction Dataset")
    data_option = st.radio(
        "Choose data source:",
        ["Use Training Dataset", "Upload New Dataset"]
    )

    if data_option == "Use Training Dataset":
        try:
            df = pd.read_csv('fraudTrain.csv')
            st.success("Training dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error loading training dataset: {str(e)}")
            return
    else:
        uploaded_file = st.file_uploader("Upload Transaction Dataset (CSV)", type=['csv'])
        if uploaded_file is None:
            st.info("Please upload a dataset to proceed.")
            return
        df = pd.read_csv(uploaded_file)

    # Display the dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Option to select a single transaction
    st.subheader("Select a Transaction for Analysis")
    selected_index = st.number_input(
        "Enter the index of the transaction you want to analyze:",
        min_value=0,
        max_value=len(df) - 1,
        step=1
    )
    selected_transaction = df.iloc[[selected_index]]

    if st.button("Analyze Selected Transaction"):
        with st.spinner("Analyzing transaction..."):
            try:
                probability, explanation = analyze_transaction(selected_transaction, model_package)
                st.markdown(f"### Fraud Probability: {probability:.1%}")
                st.markdown("#### Explanation:")
                st.write(explanation)
            except Exception as e:
                st.error(f"Error during transaction analysis: {str(e)}")

if __name__ == "__main__":
    main()
