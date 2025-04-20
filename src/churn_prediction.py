# churn_prediction.py

"""
Netflix User Churn Prediction

This script uses simulated user data to predict whether users are likely to churn.
Model: Random Forest Classifier
Outputs a classification report and a confusion matrix plot (saved to /visuals).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Make sure the visuals folder exists
os.makedirs("visuals", exist_ok=True)

# Simulate user data
np.random.seed(42)
users = pd.DataFrame({
    'avg_watch_time': np.random.normal(20, 5, 1000),  # hours/week
    'genres_watched': np.random.randint(1, 12, 1000),
    'months_inactive': np.random.randint(0, 6, 1000),
    'is_premium': np.random.randint(0, 2, 1000),
    'churned': np.random.choice([0, 1], p=[0.7, 0.3], size=1000)  # 0 = active, 1 = churned
})

# Features and labels
X = users.drop('churned', axis=1)
y = users['churned']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion matrix heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Active', 'Churned'],
            yticklabels=['Active', 'Churned'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Churn Prediction Confusion Matrix')
plt.tight_layout()

# Save confusion matrix plot
plt.savefig("visuals/churn_confusion_matrix.png")
plt.show()