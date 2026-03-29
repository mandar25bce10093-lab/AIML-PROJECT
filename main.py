import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Dataset
# Note: encoding='latin-1' is often needed for this specific CSV
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']] # Keep only label and text
df.columns = ['label', 'message']

# 2. Preprocessing
# Convert labels to numbers: ham = 0, spam = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']

# 3. Split the Data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Extraction (Convert text to numbers)
cv = CountVectorizer(decode_error='ignore')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# 5. Build and Train the Model
model = MultinomialNB()
model.fit(X_train_cv, y_train)

# 6. Evaluate
predictions = model.predict(X_test_cv)
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nDetailed Report:\n", classification_report(y_test, predictions))

# 7. Test with Your Own Message
def check_spam(msg):
    data = cv.transform([msg]).toarray()
    result = model.predict(data)
    return "SPAM 🚨" if result[0] == 1 else "HAM (Clean) ✅"

print("-" * 20)
user_msg = "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim."
print(f"Message: {user_msg}\nResult: {check_spam(user_msg)}")
