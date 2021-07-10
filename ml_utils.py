from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# define a Gaussain NB classifier
clf_gnb = GaussianNB()

#Task 3: Added a new classifier to the startup and choosing the best model accuracy

clf_ada = AdaBoostClassifier() 

clf = None

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    global clf
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf_gnb.fit(X_train, y_train)
    clf_ada.fit(X, y)


    # calculate the print the accuracy score
    acc = accuracy_score(y_test, clf_gnb.predict(X_test))
    acc_ada = accuracy_score(y_test, clf_ada.predict(X_test))

    # Comparing the accuracies of both the models and selecting the best model accuracy
    if acc <= acc_ada:
        clf = clf_ada
        print(f"Model(AdaBoostClassifier) trained with accuracy: {round(acc_ada, 3)}")
    else :
        clf = clf_gnb
        print(f"Model(GaussianNB) trained with accuracy: {round(acc, 3)}")




# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)
    # Printing out the result for debugging
    print("model has been re-trained!")
