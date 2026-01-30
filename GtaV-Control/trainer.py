import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

csv_file = "dataset.csv"
df = pd.read_csv(csv_file)

X = df.iloc[:, :-1]  # 63 features
y = df.iloc[:, -1]   # labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\n[MODEL REPORT]")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

model_file = "model.pkl"
joblib.dump(clf, model_file)
print(f"\n[SAVED] Model saved to {model_file}")
