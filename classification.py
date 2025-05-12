import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import export_graphviz
import graphviz
import seaborn as sns
import matplotlib.pyplot as plt

# Read dataset
df = pd.read_csv('data.csv')

# Data preprocessing
features = ['Start', 'End', 'G', 'U', 'bi', 'uni', 'duplex', 'Pos1', 'Pos2', 'Pos6','Pos13', 'Pos14', 'Pos18', 'Dif_5-3', 'Content+', 'Content-', 'Cons+', 'Cons-', 'Hyb19', 'target']
X = df[features]
y = df['Activity']

# binarize activity column, 0 = ineffective, 1 = effective
# if activity is less than median, activity=1. if more, activity=0
threshold = df['Activity'].median()
print(threshold)
df['Activity_Class'] = (df['Activity'] < threshold).astype(int)
y = df['Activity_Class']

print(df['Activity_Class'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression
lr_clf = LogisticRegression(random_state=42, max_iter=2000)
lr_clf.fit(X_train, y_train)
y_pred = lr_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)

# Display first decision tree
tree = rf_clf.estimators_[0]
dot_data = export_graphviz(
    tree,
    feature_names=features,
    filled=True,
    class_names=["Low Activity", "High Activity"],
    proportion=True,
    impurity=False
)
graph = graphviz.Source(dot_data)
graph.render("random_forest_tree_0", format='png')

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Random Forest Confusion Matrix")
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": features, 
    "Importance": rf_clf.feature_importances_
    }).sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance)
plt.title("Random Forest Feature Importance")
plt.show()
