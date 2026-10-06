import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Financial Sentiment Analyzer", page_icon="📈", layout="centered")

st.title("📈 Financial Sentiment Analysis App")
st.write("Single sentence mode tho patu, CSV file upload chesi **custom range** lo predictions and download chesukune facility ikkada undi!")

# Load Model and Tokenizer
@st.cache_resource
def load_artifacts():
    model = load_model('financial_sentiment_model.keras')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_artifacts()

MAX_LEN = 50
labels = ['Negative (0)', 'Neutral (1)', 'Positive (2)']

tab1, tab2 = st.tabs(["📝 Single Sentence Mode", "📁 CSV Range Prediction Mode"])

with tab1:
    user_input = st.text_area(
        "Enter Financial Text:", 
        "The corporation reported record-breaking quarterly profits that exceeded all Wall Street estimates."
    )
    
    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Dayachesi edaina text enter cheyandi!")
        else:
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
            
            prediction = model.predict(padded)
            pred_class = np.argmax(prediction, axis=1)[0]
            confidence = float(np.max(prediction)) * 100
            
            st.markdown("---")
            st.subheader("Prediction Result:")
            
            if pred_class == 2:
                st.success(f"🟢 **{labels[pred_class]}** (Confidence: {confidence:.2f}%)")
            elif pred_class == 0:
                st.error(f"🔴 **{labels[pred_class]}** (Confidence: {confidence:.2f}%)")
            else:
                st.info(f"🔵 **{labels[pred_class]}** (Confidence: {confidence:.2f}%)")

with tab2:
    st.subheader("Upload CSV File & Select Range")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        total_rows = len(df)
        st.info(f"Total rows found in CSV: **{total_rows}**")
        
        st.dataframe(df.head())
        
        text_column = st.selectbox("Select the column containing financial sentences:", df.columns)
        
        st.markdown("---")
        st.subheader("Select Row Range for Prediction")
        
        # Range selection inputs
        start_idx, end_idx = st.select_slider(
            "Select range of rows to process:",
            options=range(0, total_rows),
            value=(0, min(49, total_rows - 1))
        )
        
        st.write(f"Selected Rows: From index **{start_idx}** to **{end_idx}** (Total: {end_idx - start_idx + 1} rows)")
        
        if st.button("Run Range Prediction"):
            # Slice dataframe based on user range
            subset_df = df.iloc[start_idx : end_idx + 1].copy()
            sentences = subset_df[text_column].astype(str).tolist()
            
            # Preprocess and Predict
            seq = tokenizer.texts_to_sequences(sentences)
            padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
            
            predictions = model.predict(padded)
            pred_classes = np.argmax(predictions, axis=1)
            confidences = [float(np.max(p)) * 100 for p in predictions]
            
            # Add results to subset dataframe
            subset_df['Predicted_Sentiment'] = [labels[p] for p in pred_classes]
            subset_df['Confidence_Score'] = [f"{conf:.2f}%" for conf in confidences]
            
            st.success("Range prediction successfully complete aindi!")
            st.dataframe(subset_df)
            
            # Download button for results
            csv_output = subset_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Range Results as CSV",
                data=csv_output,
                file_name=f"sentiment_predictions_range_{start_idx}_to_{end_idx}.csv",
                mime="text/csv",
            )