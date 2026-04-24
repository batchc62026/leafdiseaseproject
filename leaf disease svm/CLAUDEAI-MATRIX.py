import joblib, numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns

clf = joblib.load("finger_vein_svm_model.joblib")
# Load your test images the same way your Pi does:
# gray → resize(64,64) → reshape(1,-1)

y_true = [...]  # actual labels from your test set
y_pred = clf.predict(X_test)

print(classification_report(y_true, y_pred,
      target_names=['Bacterial Spot','Early Blight','Healthy','Late Blight','Leaf Mold']))

sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d')
plt.show()