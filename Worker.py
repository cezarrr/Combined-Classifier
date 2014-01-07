__author__ = 'CJank'

files = ["C:\Users\CJank\Desktop\Dyskretyzator\data\\australian.dat",
         "C:\Users\CJank\Desktop\Dyskretyzator\data\\german.csv",
		 "C:\Users\CJank\Desktop\Dyskretyzator\data\\glass.dscrt",
		 "C:\Users\CJank\Desktop\Dyskretyzator\data\\Indian Liver Patient Dataset (ILPD).csv",
		 "C:\Users\CJank\Desktop\Dyskretyzator\data\\iris_number.data",
		 "C:\Users\CJank\Desktop\Dyskretyzator\data\\wine.data",
		 "C:\Users\CJank\Desktop\Dyskretyzator\data\\winequality-red.csv",
		 "C:\Users\CJank\Desktop\Dyskretyzator\data\\yeast.csv",
         "C:\Users\CJank\Desktop\Dyskretyzator\data\\australianDiscretisationResults.txt",
		 "C:\Users\CJank\Desktop\Dyskretyzator\data\\germanDiscretisationResults.txt",
		 "C:\Users\CJank\Desktop\Dyskretyzator\data\\glassDiscretisationResults.txt",
		 "C:\Users\CJank\Desktop\Dyskretyzator\data\\Indian Liver Patient Dataset (ILPD)DiscretisationResults.txt",
		 "C:\Users\CJank\Desktop\Dyskretyzator\data\\iris_numberDiscretisationResults.txt",
		 "C:\Users\CJank\Desktop\Dyskretyzator\data\\wineDiscretisationResults.txt",
		 "C:\Users\CJank\Desktop\Dyskretyzator\data\\winequality-redDiscretisationResults.txt",
		 "C:\Users\CJank\Desktop\Dyskretyzator\data\\yeastDiscretisationResults.txt",
        ]


import Loader as ldr
import combinedClassifier as cmb
import numpy as np
import csv
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
import ntpath

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
fileNames =[]
for path in files:
    fileNames.append(path_leaf(path))


names = ["Nearest Neighbor 1","Nearest Neighbor 3",
         "Nearest Neighbor 5", "Nearest Neighbor 7",
         "Nearest Neighbor 9",
         #"Linear SVM",
         "RBF SVM",
         "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA",
         #"QDA"
]
classifiers = [
	KNeighborsClassifier(1),
    KNeighborsClassifier(3),
	KNeighborsClassifier(5),
    KNeighborsClassifier(7),
    KNeighborsClassifier(9),
    #SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    #QDA()
]

combinedClf = cmb.CombinedClassifier(classifiers)

wholeDataSet=np.array(ldr.loadData("C:\Users\CJank\Desktop\Dyskretyzator\data\\iris_number.data"))
X = wholeDataSet[:,0:-1]
y = wholeDataSet[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
combinedClf.fit(X_train,y_train)
results = combinedClf.predict(X_train)
CVresults = cross_val_score(combinedClf, X, y, cv=5)
CVres = np.array(CVresults).mean()
print CVres


resultsOfCV=[]
for ds in files:
    wholeDataSet=np.array(ldr.loadData(ds))
    X = wholeDataSet[:,0:-1]
    y = wholeDataSet[:,-1]

    print ds

    cv_p = 5

    tmpRes = cmb.crossValidManyClassifiers(classifiers,X,y,cv_p);
    resultsOfCV.extend(tmpRes)



    #for name,clf in zip(names,classifiers):


    #for name, clf in zip(names, classifiers):
    #
    #
    #
    #    #clf.fit(X_train,y_train)
    #    #score = clf.score(X_test, y_test)
    #    #score_train = clf.score(X_train, y_train)
    #
    #    scores = cross_val_score(clf, X, y, cv=5)
    #    CVscore = np.mean(scores)
    #
    #    #print name+"-> "+str(score)
    #    #print name+"-> "+str(score_train)+" (on training set)"
    #    print name+"-> "+str(CVscore)#+" (on CV-10)"

resultsOfCV=[x*100 for x in resultsOfCV]
resultsArray = np.array(resultsOfCV)
resultsArray=resultsArray.reshape(files.__len__(),classifiers.__len__())
bestResult = resultsArray.max(axis=1)
bestClassifier = resultsArray.argmax(axis=1)
bestClassifier = [names[clIdx] for clIdx in bestClassifier]

with open("C:\Users\CJank\Desktop\\test.csv", 'wb') as csvFile:
    csvWriter = csv.writer(csvFile, delimiter=',')
    csvWriter.writerow([" "]+names+['','Najlepszy klasyfikator', 'Najlepsza klasyfikacja'])
    j=classifiers.__len__()
    for i in range (fileNames.__len__()):
        subList=resultsArray[i,:].tolist()
        csvWriter.writerow([fileNames[i]]+subList+['',bestClassifier[i],bestResult[i]])


print "Done!"