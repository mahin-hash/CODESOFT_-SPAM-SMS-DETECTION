# 📩 Spam SMS Detection using Machine Learning
A machine learning project to classify SMS messages as **spam** or **ham (legitimate)** using natural language processing and traditional ML classifiers. This project uses **TF-IDF vectorization** and compares three popular models:
- ✅ Logistic Regression  
- ✅ Naive Bayes  
- ✅ Support Vector Machine (SVM)

## 🧠 Objective
To build a robust text classification model that can:
- Automatically detect spam messages.
- Compare performance across multiple ML algorithms.
- Visualize model accuracy and confusion matrices.

## 🛠️ Technologies Used
- **Python 3**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**

## ⚙️ Setup Instructions

1. **Clone or download this repo**
2. Place the dataset file (`spam.csv`) in the same folder as the script
3. Install dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn
```
4. Run the model script:
```bash
python spam_sms_detection.py
```
## 🔍 How It Works
1. TF-IDF Vectorization of the text messages

2. Training of 3 classifiers:

Logistic Regression

Naive Bayes

SVM (LinearSVC)

3. Evaluation using:

Accuracy

Confusion Matrix

Classification Report

4. Visualization of:

Model accuracy comparison (bar chart)

Confusion matrix heatmaps

4. Prediction of a custom sample SMS message

## 📊 Sample Output
Logistic Regression Accuracy: 0.982
Naive Bayes Accuracy: 0.965
SVM Accuracy: 0.984
📩 Sample Message:
"Congratulations! You've won a $1000 Walmart gift card. Call now!"

→ Predicted as Spam by all models.

## 📌 Folder Structure
```bash
├── spam_sms_detection.py
├── spam.csv
└── README.md
```
