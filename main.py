# main.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Dataset
file_path = 'data/archive/kddcup.data_10_percent_corrected'  # Make sure file is exactly here
data = pd.read_csv(file_path, header=None)

# Step 2: Add Column Names
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]
data.columns = columns

# Step 3: Basic Cleaning
data = data.drop_duplicates()
data = data.reset_index(drop=True)

# Step 4: Encode Categorical Columns
cat_cols = ["protocol_type", "service", "flag"]
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Step 5: Label Binary (normal = 0, attack = 1)
data['label'] = data['label'].apply(lambda x: 0 if x == 'normal.' else 1)

# Step 6: Train-Test Split
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 7: Apply Isolation Forest
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)

# Step 8: Predict Anomalies
y_pred = model.predict(X_test)

# Step 9: Convert predictions to 0/1
y_pred = [0 if x == 1 else 1 for x in y_pred]

# Step 10: Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Visualize with heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).iloc[:-1, :-1], annot=True)
plt.title("Classification Report Heatmap")
plt.show()
