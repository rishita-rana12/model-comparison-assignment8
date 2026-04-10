import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Models
lr = LogisticRegression(max_iter=1000)
svm = SVC(probability=True)
rf = RandomForestClassifier()

# Train
lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Probabilities
lr_prob = lr.predict_proba(X_test)[:,1]
svm_prob = svm.predict_proba(X_test)[:,1]
rf_prob = rf.predict_proba(X_test)[:,1]

# ROC
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_prob)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

# AUC
lr_auc = auc(lr_fpr, lr_tpr)
svm_auc = auc(svm_fpr, svm_tpr)
rf_auc = auc(rf_fpr, rf_tpr)

# Plot
plt.plot(lr_fpr, lr_tpr, label="LR AUC="+str(lr_auc))
plt.plot(svm_fpr, svm_tpr, label="SVM AUC="+str(svm_auc))
plt.plot(rf_fpr, rf_tpr, label="RF AUC="+str(rf_auc))

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Model Comparison ROC")
plt.legend()
plt.show()

print("LR AUC:", lr_auc)
print("SVM AUC:", svm_auc)
print("RF AUC:", rf_auc)