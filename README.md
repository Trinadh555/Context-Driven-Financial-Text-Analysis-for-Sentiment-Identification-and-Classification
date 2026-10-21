# Context-Driven Financial Text Analysis for Sentiment Identification and Classification

An advanced Natural Language Processing (NLP) deep learning project designed to analyze and classify the sentiment of financial texts into Negative, Neutral, and Positive categories using a Fine-Tuned Bidirectional LSTM (BiLSTM) model.

## Problem Statement
Financial markets are heavily influenced by news, reports, and social media. Accurately extracting the underlying sentiment from financial texts is crucial for market trend analysis, algorithmic trading, and investment decision-making. 

The objective of this project is to develop a robust Deep Learning sequence model (BiLSTM) capable of understanding the context of financial terminology and accurately classifying the sentiment, even when dealing with imbalanced real-world datasets.

## Dataset
The dataset comprises financial sentences and headlines labeled with their corresponding sentiments.
- **Total Samples Loaded:** 5,836
- **Data Split:**
  - **Training Set (70%):** 4,084 samples
  - **Validation Set (10%):** 584 samples
  - **Testing Set (20%):** 1,168 samples

### Class Distribution & Imbalance Handling
The dataset is naturally imbalanced, with a majority of neutral financial statements:
- `Neutral (1)`: 3,124 samples
- `Positive (2)`: 1,852 samples
- `Negative (0)`: 860 samples

To prevent the model from becoming biased towards the majority class, **Class Weights** were computed and applied during training:
- `Negative (0)`: 2.26
- `Neutral (1)`: 0.62
- `Positive (2)`: 1.05

## Exploratory Data Analysis & Feature Correlation
Basic NLP feature engineering was performed, extracting attributes like `Sentence_Length` and `Word_Count`. 
The correlation matrix below indicates that while word count and sentence length are highly correlated with each other, they have a near-zero linear correlation with the sentiment `label` itself. This justifies the need for complex, sequence-based Deep Learning models rather than simple linear classifiers.

![Feature Correlation Matrix](assets/image_5ec7fe.png)

## Deep Learning Model: Fine-Tuned BiLSTM
A Bidirectional Long Short-Term Memory (BiLSTM) network was selected for this task. BiLSTMs are highly effective for text classification because they process the sequential text data in both forward and backward directions, capturing the full context of a word within a sentence.

- **Optimizer:** Adam (Learning Rate: 0.001)
- **Loss Function:** Sparse Categorical Crossentropy (with class weights)
- **Early Stopping Triggered:** The model trained effectively, reaching an optimal state around Epoch 7 with a training accuracy of ~87%.

## Model Performance
The model was evaluated on the unseen Testing Set (1,168 samples) and achieved an overall **Test Accuracy of 73%**.

### Classification Report

| Sentiment Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Negative (0)** | 0.44 | 0.53 | 0.48 | 172 |
| **Neutral (1)** | 0.79 | 0.76 | 0.78 | 625 |
| **Positive (2)** | 0.78 | 0.75 | 0.76 | 371 |
| *Macro Avg* | *0.67* | *0.68* | *0.67* | *1168* |
| *Weighted Avg* | *0.74* | *0.73* | *0.73* | *1168* |

*Note: The model performs exceptionally well on Neutral and Positive texts. Negative sentiment detection has lower precision due to the high class imbalance and nuanced financial language, but the recall of 0.53 shows the class weights successfully forced the model to identify negative instances.*

### Visualizing Performance

**Model Accuracy & Loss Progress (Training vs Validation):**
*(Demonstrates the learning curve and stabilization across epochs)*
![Training Curves](assets/image_5e7642.png)

**Confusion Matrix:**
*(Shows the true vs. predicted classifications across the test set)*
![Confusion Matrix](assets/image_5ec7be.png)

## Project Structure
```text
Financial-Sentiment-Analysis-BiLSTM/
│
├── app.py                     # Web application for live text analysis (if applicable)
├── requirements.txt           # Python library dependencies
├── model_performance_report.csv # Exported classification metrics
├── bilstm_model.h5            # Saved fine-tuned BiLSTM model weights
├── README.md                  # Project documentation
└── assets/                    # Project images and charts

git clone [https://github.com/Trinadh555/Context-Driven-Financial-Text-Analysis-for-Sentiment-Identification-and-Classification.git](https://github.com/Trinadh555/Context-Driven-Financial-Text-Analysis-for-Sentiment-Identification-and-Classification.git)
cd Context-Driven-Financial-Text-Analysis-for-Sentiment-Identification-and-Classification

pip install -r requirements.txt

python app.py
