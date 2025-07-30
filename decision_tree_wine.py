import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

data = pd.read_csv("winequality-red.csv", sep=",", header=0)

data.columns = data.columns.str.strip()

print("Missing values in each column:")
print(data.isnull().sum())


X = data.drop("quality", axis=1)
y = data["quality"]

y = y.apply(lambda q: 1 if q >= 7 else 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(20, 10))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=["Bad", "Good"])
plt.title("Decision Tree - Wine Quality")
plt.show()
