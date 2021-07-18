
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Define all classification model 
clf = RandomForestClassifier(n_estimators=100, random_state=0)


# final outcome class
classes = {
    True : 'defects',
    False : 'No defects'
}
r_classes = {y: x for x, y in classes.items()}
# function to load_model for Naive - Baiye's
def load_model():
        data = pd.read_csv('bug_pred.csv')
        X = data.drop('defects', axis = 1)
        y = data['defects']

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
        clf.fit(X_train, y_train)

        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"Model trained with accuracy: {round(acc, 3)}")

def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])[0] 
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [classes[d.defects] for d in data]
    #fit the classifier again based on the new data obtained
    clf.fit(X,y)


