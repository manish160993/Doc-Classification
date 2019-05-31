'''File to train and save Classical ML model''' 
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def calculate_confidence_interval(model, x, y, CV):
    '''function to calculate the confidence interval of a classification made. 
    I do not know if accuracy is normally distributed for this data, so I am going to 
    calcuate confidence interval using bootstrap resampling methods'''
    accuracies = cross_val_score(model,x,y,scoring='accuracy',cv=CV) #get accuracies from bootstrapped test data
    mean_acc = accuracies.mean()
    bounds_acc = accuracies.std() * 2
    print(f"Accuracy: {mean_acc}% +/- {bounds_acc}% with 95% likelihood")

def get_data():
    x_train = pd.read_csv("data/x_train.csv", header=0)
    y_train = pd.read_csv("data/y_train.csv", header=0)
    x_test = pd.read_csv("data/x_test.csv", header=0)
    y_test = pd.read_csv("data/y_test.csv", header=None)
    x_val = pd.read_csv("data/x_val.csv", header=0)
    y_val = pd.read_csv("data/y_val.csv", header=None)
    return x_train, y_train, x_test, y_test, x_val, y_val

random_seed = 0 # should be same across all files
x_train, y_train, x_test, y_test, x_val, y_val = get_data()
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=random_seed),
    MultinomialNB(),
    LogisticRegression(random_state=random_seed),
    #SVC(gamma='auto'),
    #KNeighborsClassifier(n_neighbors=3)
]

# Experimentation
CV = 5
for model in models:
    model_name = models.__class__.__name__
    accuracies = cross_val_score(model,x_train,y_train,scoring='accuracy',cv=CV)
    print(model_name, accuracies)

# on best model: LR
from sklearn.metrics import confusion_matrix
clf = LogisticRegression(random_state=random_seed).fit(x_train,y_train)
# save model
pickle.dump(clf, open('saved_data_objects/LRModel.pkl', 'wb'))
y_pred = clf.predict(x_val)
print("Accuracy on Val set: ", clf.score(x_val,y_val))
print("Confusion Matrix for validation data set: ")
print(confusion_matrix(y_val, y_pred))
calculate_confidence_interval(clf,x_train,y_train,10)

