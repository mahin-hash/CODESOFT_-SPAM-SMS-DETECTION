import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load and clean data
df = pd.read_csv("spam.csv", encoding="ISO-8859-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Step 3: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Define and train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC()
}

accuracies = {}
conf_matrices = {}

for name, model in models.items():
    print(f"\n🔎 Training {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    accuracies[name] = acc
    conf_matrices[name] = cm

    print(f"\n✅ {name} Results")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 5: Plot Accuracy Comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette="viridis")
plt.ylabel("Accuracy")
plt.ylim(0.9, 1.0)
plt.title("📊 Model Accuracy Comparison")
plt.tight_layout()
plt.show()

# Step 6: Plot Confusion Matrices
for name, cm in conf_matrices.items():
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f"🔍 Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# Step 7: Predict a sample message with all models
sample = ["Congratulations! You've won a $1000 Walmart gift card. Call now!"]
sample_tfidf = vectorizer.transform(sample)

print("\n📩 Sample Message Prediction:")
for name, model in models.items():
    pred = model.predict(sample_tfidf)[0]
    print(f"{name}: {'Spam' if pred == 1 else 'Ham'}")
