import pandas as pd
import os
import pickle
df = pd.read_csv(
	"data/input/iris.data",
	header = None,
	names = ["feat1","feat2","feat3","feat4","label"])

X = df[["feat1","feat2","feat3","feat4"]].values
y = df[["label"]].values.ravel()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y)

from sklearn.svm import SVC

clf = SVC()
clf.fit(X_train, y_train)

y_predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_predictions)

model_path = "data/output/model.pkl"
pickle.dump(clf, open(model_path, 'wb'))
