# 🎭 Emotion Detection using NLP

A machine learning project that classifies the **emotion behind a piece of text** — joy, sadness, anger, fear, love, or surprise — using classic NLP techniques and an interactive Streamlit web app.

![UI Preview](assets/screenshot.png)
<!-- Add a screenshot of your running app above. Save it as assets/screenshot.png in this repo. -->

## 🚀 Demo

Run it locally in under a minute — see [Getting Started](#-getting-started) below.

Overview

This project compares multiple text classification pipelines on the [Emotions dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions) (~16,000 labeled sentences) and ships the best-performing one behind a polished Streamlit UI.

| Model | Features | Accuracy |

 Naive Bayes | Bag of Words | 77% |
 Multinomial Naive Bayes | TF-IDF | 80% |
 Logistic Regression on Multinomial Naive Bayes | TF-IDF | 86% |


Emotion classes: `joy` · `sadness` · `anger` · `fear` · `love` · `surprise`

##  How it works

1. **Preprocessing** — lowercasing, punctuation removal, number removal, emoji/non-ASCII removal, stopword removal (NLTK)
2. **Feature extraction** — TF-IDF vectorization
3. **Classification** — Logistic Regression
4. The vectorizer + model are bundled into a single `scikit-learn` **Pipeline**, saved as `emo_model.pkl`
5. A Streamlit app (`app.py`) loads that pipeline and predicts emotion + confidence scores for any text you type in

## 📁 Project Structure

```
.
├── app.py                 # Streamlit UI
├── preprocessing.py       # Text cleaning used by both training and the app
├── nlp.ipynb              # Model training & experimentation notebook
├── emo_model.pkl          # Trained pipeline (TF-IDF + Logistic Regression)
├── train.txt              # Dataset (text;emotion format)
└── README.md
```


Tech Stack & Techniques 

Used Language: Python
Libraries: Pandas, NumPy, NLTK / Scikit-learn 
NLP Techniques: Tokenization, Stop-word Removal, Stemming
Vectorization: TF-IDF / Bag of Words (Count-Vectorizer) 
ML Model: Naive Bayes /Multinomial Naive Bayes /Logistic Regression
Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix 
Deployment: Streamlit

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.



If you want to reproduce or retrain the model from `train.txt`, open and run `nlp.ipynb`.

## Model Performance

The final Logistic Regression + TF-IDF model performs strongly on the majority classes (`joy`, `sadness`) but has lower recall on underrepresented classes (`love`, `surprise`) due to class imbalance in the dataset. Future improvements could include:

- `class_weight="balanced"` in Logistic Regression
- Oversampling minority classes (SMOTE)
- Trying transformer-based embeddings (e.g. DistilBERT)

## 🖼️ Screenshot

<!-- Replace this with an actual screenshot of your app -->
 <img width="1537" height="865" alt="EMO_UI" src="https://github.com/user-attachments/assets/b386359b-ce80-4175-8218-ae744610da31" />
 <img width="1107" height="715" alt="Screenshot 2026-09-26 140220" src="https://github.com/user-attachments/assets/ad4522bf-35ff-4bbe-9d79-53398b70b1d3" />
 <img width="1573" height="850" alt="RESULT_UI" src="https://github.com/user-attachments/assets/1158e36c-5263-4178-88ef-e186260f456f" />
 <img width="1287" height="852" alt="Screenshot 2026-09-26 135729" src="https://github.com/user-attachments/assets/f26db5f7-8311-46f0-a106-5f019f6a8c0f" />
 <img width="1150" height="867" alt="Screenshot 2026-09-26 135906" src="https://github.com/user-attachments/assets/f1d38822-668c-4041-a5c8-7eec9184a682" />
 <img width="1192" height="832" alt="Screenshot 2026-09-26 135956" src="https://github.com/user-attachments/assets/0c61b9c3-987d-4891-b0b1-8492b61b27ff" />

 
 

 ("C:\Users\ABHISHEK\Pictures\Screenshots\RESULT_UI.png") 


## 🙋 Author

Built by **Abhishek Negi ** as an NLP / ML  project.
