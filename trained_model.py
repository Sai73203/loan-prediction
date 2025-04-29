import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
df = pd.read_csv("loan_data.csv")

# Preprocessing
df.dropna(inplace=True)
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    df[col] = le.fit_transform(df[col])

X = df[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Education', 'Self_Employed', 'Property_Area']]
y = df['Loan_Status']

# Check class distribution before SMOTE
print("Original Class Distribution:\n", y.value_counts())

# Balance dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("\nAfter SMOTE Class Distribution:\n", pd.Series(y_resampled).value_counts())


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test)


# Accuracy & classification report
acc = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", acc)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\n🧮 Confusion Matrix:\n", cm)

# Heatmap of confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# Save model
pickle.dump(model, open("loan_model.pkl", "wb")) #it uses two parametes ; that is model object and open the empty file of pkl with write(wb).

# dump means it creates the duplicate data (or) dummie data.
