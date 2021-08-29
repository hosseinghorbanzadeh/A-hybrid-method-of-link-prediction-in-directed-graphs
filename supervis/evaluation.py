import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
from operator import itemgetter, attrgetter
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter
from itertools import islice
import operator
from networkx.algorithms import community
import matplotlib.pyplot as pld
from sklearn.neighbors import KNeighborsClassifier
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.preprocessing import normalize
from scipy import interp
from sklearn.model_selection import StratifiedKFold
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
from sklearn import svm, datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import *
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z
def common_out_neighbors(g, i, j):
    return set(g.successors(i)).intersection(g.successors(j))

def common_in_neighbors(g, i, j):
    return set(g.predecessors(i)).intersection(g.predecessors(j))

def roc_com_Measure(data1,nb_clf,str1,strNameDataSet):
    def roc_curve_and_score(y_test, pred_proba):
        fpr, tpr, _ = roc_curve(y_test.ravel(), pred_proba.ravel())
        roc_auc = roc_auc_score(y_test.ravel(), pred_proba.ravel())
        return fpr, tpr, roc_auc
    
    Y=data1.iloc[:,[17]]
    print(Y)
    y=(Y.values.ravel())

    X=data1.iloc[:,[0,1,13]]

    #SCNHA
    X=data1.iloc[:,[0,1,11]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=17)
    nb_clf.fit(X_train, y_train)
    nb_prediction_proba = nb_clf.predict_proba(X_test)[:, 1]
    
    fpr, tpr, roc_auc = roc_curve_and_score(y_test, nb_prediction_proba)
    plt.plot(fpr, tpr, color='red', lw=2,
         label='ROC AUC SCNHA={0:.3f}'.format(roc_auc))

    X=data1.iloc[:,[0,1,15]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=17)
    nb_clf.fit(X_train, y_train)
    nb_prediction_proba = nb_clf.predict_proba(X_test)[:, 1]
    
    fpr, tpr, roc_auc = roc_curve_and_score(y_test, nb_prediction_proba)
    plt.plot(fpr, tpr, color='pink', lw=2,
         label='ROC AUC N2V={0:.3f}'.format(roc_auc))

    X=data1.iloc[:,[0,1,8]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=17)
    nb_clf.fit(X_train, y_train)
    nb_prediction_proba = nb_clf.predict_proba(X_test)[:, 1]
    
    fpr, tpr, roc_auc = roc_curve_and_score(y_test, nb_prediction_proba)
    plt.plot(fpr, tpr, color='green', lw=2,
         label='ROC AUC RAinout={0:.3f}'.format(roc_auc))

    X=data1.iloc[:,[0,1,5]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=17)
    nb_clf.fit(X_train, y_train)
    nb_prediction_proba = nb_clf.predict_proba(X_test)[:, 1]
    
    fpr, tpr, roc_auc = roc_curve_and_score(y_test, nb_prediction_proba)
    plt.plot(fpr, tpr, color='blue', lw=2,
         label='ROC AUC CNinout={0:.3f}'.format(roc_auc))

    X=data1.iloc[:,[0,1,4]]
    X=data1.iloc[:,[0,1,3]]
    X=data1.iloc[:,[0,1,2]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=17)
    nb_clf.fit(X_train, y_train)
    nb_prediction_proba = nb_clf.predict_proba(X_test)[:, 1]
    
    fpr, tpr, roc_auc = roc_curve_and_score(y_test, nb_prediction_proba)
    plt.plot(fpr, tpr, color='black', lw=2,
         label='ROC AUC AAinout={0:.3f}'.format(roc_auc))
    
    X=data1.iloc[:,[0,1,14]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=17)
    nb_clf.fit(X_train, y_train)
    nb_prediction_proba = nb_clf.predict_proba(X_test)[:, 1]
    
    fpr, tpr, roc_auc = roc_curve_and_score(y_test, nb_prediction_proba)
    plt.plot(fpr, tpr, color='yellow', lw=2,
         label='ROC AUC Triadic={0:.3f}'.format(roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('network :'+strNameDataSet +'-  Classifier :'+str1)
    plt.show()
    
    

def K_Fold_ROC_AUC(X,y,classifier):    
    #X=Dataframe.iloc[:,[0,1,5]]
    #y=Dataframe.iloc[:,[6]]
    y=(y.values.ravel())
    X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape

    # Add noisy features
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 2 * n_features)]
    cv = StratifiedKFold(n_splits=4)
    #classifier = svm.SVC(kernel='linear', probability=True,
                     #random_state=random_state)
    #classifier =LogisticRegression(random_state=0, solver='lbfgs',max_iter=1000000)
    #classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                     #max_depth=1, random_state=0)
   
    alpha = 0.95
 
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        #classifier.fit(X[train], y[train])
        #probas_=classifier.predict_proba(X[test])
        #print(y[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        #print(roc_auc)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
        #print('ok')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC LogisticRegression Model for HALP PRLP...')
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc

    
    

def Confucion_Matrix1(X,Y,Metod,classifier):
    print('---------------',Metod,'--------------')
    alpha = 0.5
    
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=42,stratify=Y)
    classifier.fit(X_train,Y_train.values.ravel())
    PredictLink=classifier.predict(X_test)
    
    y_pred_prob = classifier.predict_proba(X_test)[:,1]
    
    Y_test1=Y_test.values.ravel()
    
    TP=0
    FN=0
    FP=0
    TN=0
    
    for i in range(len(Y_test)):
        #print(Y_test1[i],PredictLink[i])
        if((Y_test1[i]==1)and(PredictLink[i]==1)):
            TP+=1
        if((Y_test1[i]==1)and(PredictLink[i]==0)):
            FN+=1
        if((Y_test1[i]==0)and(PredictLink[i]==1)):
            FP+=1
        if((Y_test1[i]==0)and(PredictLink[i]==0)):
            TN+=1
    Accuracy=(TP+TN)/(TP+FN+FP+TN)
    print('Accuracy',Accuracy)
    precision=(TP)/(TP+FP)
    print('precision',precision)
    print('AUC',roc_auc_score(Y_test, y_pred_prob))
    #print(confusion_matrix(Y_test,PredictLink,[1,0]))
    Total=(TP+FN+FP+TN)
    Random_Accuracy=(((TN+FP)*(TN+FN))+((FN+TP)*(FP+TP)))/(Total*Total)
    Kappa=(Accuracy-Random_Accuracy)/(1-Random_Accuracy)
    print('Kappa',Kappa)
    #print(confusion_matrix(Y_test,PredictLink))
    print(classification_report(Y_test,PredictLink,[1,0]))
    from sklearn import metrics

    

def Confucion_Matrix(X,y,Metod,classifier):
    from sklearn import metrics
    print('---------------',Metod,'--------------')
    ListACC=[]
    ListKappa=[]
    ListPre=[]
    ListAuc=[]
    #ListAuc=K_Fold_ROC_AUC(X,y,classifier)
    cv = StratifiedKFold(n_splits=4)
    #kf = KFold(n_splits=4)
    y=(y.values.ravel())
    X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape

    # Add noisy features
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 2 * n_features)]
    for train, test in cv.split(X,y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train], X[test]
        Y_train, Y_test = y[train], y[test]
        classifier.fit(X[train], y[train])
        PredictLink=classifier.predict(X_test)
        TP=0
        FN=0
        FP=0
        TN=0
        y_pred_prob = classifier.predict_proba(X_test)[:,1]
        #Y_test1=Y_test
        for i in range(len(Y_test)):
            #print(Y_test1[i],PredictLink[i])
            if((Y_test[i]==1)and(PredictLink[i]==1)):
                TP+=1
            if((Y_test[i]==1)and(PredictLink[i]==0)):
                FN+=1
            if((Y_test[i]==0)and(PredictLink[i]==1)):
                FP+=1
            if((Y_test[i]==0)and(PredictLink[i]==0)):
                TN+=1
        Accuracy=(TP+TN)/(TP+FN+FP+TN)
        ListACC.append(Accuracy)
        #print('Accuracy',Accuracy)
        precision=(TP)/(TP+FP)
        ListPre.append(precision)
        #print('precision',precision)
        #print('AUC',roc_auc_score(Y_test, y_pred_prob))
        ListAuc.append(roc_auc_score(Y_test, y_pred_prob))
        #print(confusion_matrix(Y_test,PredictLink,[1,0]))
        Total=(TP+FN+FP+TN)
        Random_Accuracy=(((TN+FP)*(TN+FN))+((FN+TP)*(FP+TP)))/(Total*Total)
        Kappa=(Accuracy-Random_Accuracy)/(1-Random_Accuracy)
        ListKappa.append(Kappa)
    print('Accuracy :',np.mean(ListACC))
    print('precision :',np.mean(ListPre))
    print('AUC :',np.mean(ListAuc))
    print('Kappa :',np.mean(ListKappa))
    w1=round(np.mean(ListAuc),4)
    return(w1)
       
    
    
def Compear_Index(data1,Str):
    print('###########################################',Str,'##############################################')
    print(data1)
    ResultAUC_Dic={}
    dictAUC={}
    Y=data1.iloc[:,[17]]
    print('____________________________','LogisticRegression','____________________________________________')
    classifier =LogisticRegression(random_state=0, solver='lbfgs',max_iter=100000)
    roc_com_Measure(data1,classifier,'LogisticRegression',Str)
    
    X=data1.iloc[:,[0,1,13]]
    ListAuc=Confucion_Matrix(X,Y,'CNAH',classifier)
    ResultAUC_Dic.update({'CNAH':ListAuc})
    
    X=data1.iloc[:,[0,1,12]]
    ListAuc=Confucion_Matrix(X,Y,'CNHA',classifier)
    ResultAUC_Dic.update({'CNHA':ListAuc})
    
    X=data1.iloc[:,[0,1,11]]
    ListAuc=Confucion_Matrix(X,Y,'CNHA-AH',classifier)
    ResultAUC_Dic.update({'CNHA-AH':ListAuc})
    
    X=data1.iloc[:,[0,1,10]]
    ListAuc=Confucion_Matrix(X,Y,'RA-Out',classifier)
    ResultAUC_Dic.update({'RA-Out':ListAuc})
    
    X=data1.iloc[:,[0,1,9]]
    ListAuc=Confucion_Matrix(X,Y,'RA-in',classifier)
    ResultAUC_Dic.update({'RA-in':ListAuc})
    
    X=data1.iloc[:,[0,1,8]]
    ListAuc=Confucion_Matrix(X,Y,'RA-in-out',classifier)
    ResultAUC_Dic.update({'RA-in-Out':ListAuc})

    X=data1.iloc[:,[0,1,7]]
    ListAuc=Confucion_Matrix(X,Y,'CND-in',classifier)
    ResultAUC_Dic.update({'CN-in':ListAuc})

    X=data1.iloc[:,[0,1,6]]
    ListAuc=Confucion_Matrix(X,Y,'CND-out',classifier)
    ResultAUC_Dic.update({'CN-out':ListAuc})

    X=data1.iloc[:,[0,1,5]]
    ListAuc=Confucion_Matrix(X,Y,'CN-inout',classifier)
    ResultAUC_Dic.update({'CN-inout':ListAuc})
    
    X=data1.iloc[:,[0,1,4]]
    ListAuc=Confucion_Matrix(X,Y,'AA-Out',classifier)
    ResultAUC_Dic.update({'AA-Out':ListAuc})
    
    X=data1.iloc[:,[0,1,3]]
    ListAuc=Confucion_Matrix(X,Y,'AA-in',classifier)
    ResultAUC_Dic.update({'AA-in':ListAuc})

    X=data1.iloc[:,[0,1,2]]
    ListAuc=Confucion_Matrix(X,Y,'AA-in-out',classifier)
    ResultAUC_Dic.update({'AA-in-Out':ListAuc})
    
    X=data1.iloc[:,[0,1,14]]
    ListAuc=Confucion_Matrix(X,Y,'Triadic',classifier)
    print(ListAuc)
    ResultAUC_Dic.update({'Triadic':ListAuc})

    X=data1.iloc[:,[0,1,15]]
    ListAuc=Confucion_Matrix(X,Y,'N2V',classifier)
    ResultAUC_Dic.update({'N2V':ListAuc})
    
   
    dictAUC.update({'LogisticRegression':ResultAUC_Dic})
    print(dictAUC)
    
    
    #K_Fold_ROC_AUC(X,Y,classifier)
    #X=data1.iloc[:,[0,1,5]]
    #K_Fold_ROC_AUC(X,Y,classifier)
    print('____________________________','LogisticRegression','____________________________________________')
    print('____________________________','GradientBoostingClassifier','____________________________________')
    alpha = 0.95
    
    ResultAUC_Dic={}
    
    classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                     max_depth=1, random_state=0)
    X=data1.iloc[:,[0,1,13]]
    ListAuc=Confucion_Matrix(X,Y,'CNAH',classifier)
    ResultAUC_Dic.update({'CNAH':ListAuc})
    
    X=data1.iloc[:,[0,1,12]]
    ListAuc=Confucion_Matrix(X,Y,'CNHA',classifier)
    ResultAUC_Dic.update({'CNHA':ListAuc})
    
    X=data1.iloc[:,[0,1,11]]
    ListAuc=Confucion_Matrix(X,Y,'CNHA-AH',classifier)
    ResultAUC_Dic.update({'CNHA-AH':ListAuc})
    
    X=data1.iloc[:,[0,1,10]]
    ListAuc=Confucion_Matrix(X,Y,'RA-Out',classifier)
    ResultAUC_Dic.update({'RA-Out':ListAuc})
    
    X=data1.iloc[:,[0,1,9]]
    ListAuc=Confucion_Matrix(X,Y,'RA-in',classifier)
    ResultAUC_Dic.update({'RA-in':ListAuc})
    
    X=data1.iloc[:,[0,1,8]]
    ListAuc=Confucion_Matrix(X,Y,'RA-in-out',classifier)
    ResultAUC_Dic.update({'RA-in-Out':ListAuc})

    X=data1.iloc[:,[0,1,7]]
    ListAuc=Confucion_Matrix(X,Y,'CND-in',classifier)
    ResultAUC_Dic.update({'CN-in':ListAuc})

    X=data1.iloc[:,[0,1,6]]
    ListAuc=Confucion_Matrix(X,Y,'CND-out',classifier)
    ResultAUC_Dic.update({'CN-out':ListAuc})

    X=data1.iloc[:,[0,1,5]]
    ListAuc=Confucion_Matrix(X,Y,'CN-inout',classifier)
    ResultAUC_Dic.update({'CN-inout':ListAuc})
    
    X=data1.iloc[:,[0,1,4]]
    ListAuc=Confucion_Matrix(X,Y,'AA-Out',classifier)
    ResultAUC_Dic.update({'AA-Out':ListAuc})
    
    X=data1.iloc[:,[0,1,3]]
    ListAuc=Confucion_Matrix(X,Y,'AA-in',classifier)
    ResultAUC_Dic.update({'AA-in':ListAuc})

    X=data1.iloc[:,[0,1,2]]
    ListAuc=Confucion_Matrix(X,Y,'AA-in-out',classifier)
    ResultAUC_Dic.update({'AA-in-Out':ListAuc})
    
    X=data1.iloc[:,[0,1,14]]
    ListAuc=Confucion_Matrix(X,Y,'Triadic',classifier)
    ResultAUC_Dic.update({'Triadic':ListAuc})

    X=data1.iloc[:,[0,1,15]]
    ListAuc=Confucion_Matrix(X,Y,'N2V',classifier)
    ResultAUC_Dic.update({'N2V':ListAuc})
    
    dictAUC.update({'GradientBoostingClassifier':ResultAUC_Dic})
    print(dictAUC)
    print('____________________________','GradientBoostingClassifier','____________________________________')
    print('____________________________','LinearDiscriminantAnalysis','____________________________________')

    classifier =  LinearDiscriminantAnalysis()
    
    ResultAUC_Dic={}
    roc_com_Measure(data1,classifier,'LinearDiscriminant',Str)
    X=data1.iloc[:,[0,1,13]]
    ListAuc=Confucion_Matrix(X,Y,'CNAH',classifier)
    ResultAUC_Dic.update({'CNAH':ListAuc})
    
    X=data1.iloc[:,[0,1,12]]
    ListAuc=Confucion_Matrix(X,Y,'CNHA',classifier)
    ResultAUC_Dic.update({'CNHA':ListAuc})
    
    X=data1.iloc[:,[0,1,11]]
    ListAuc=Confucion_Matrix(X,Y,'CNHA-AH',classifier)
    ResultAUC_Dic.update({'CNHA-AH':ListAuc})
    
    X=data1.iloc[:,[0,1,10]]
    ListAuc=Confucion_Matrix(X,Y,'RA-Out',classifier)
    ResultAUC_Dic.update({'RA-Out':ListAuc})
    
    X=data1.iloc[:,[0,1,9]]
    ListAuc=Confucion_Matrix(X,Y,'RA-in',classifier)
    ResultAUC_Dic.update({'RA-in':ListAuc})
    
    X=data1.iloc[:,[0,1,8]]
    ListAuc=Confucion_Matrix(X,Y,'RA-in-out',classifier)
    ResultAUC_Dic.update({'RA-in-Out':ListAuc})

    X=data1.iloc[:,[0,1,7]]
    ListAuc=Confucion_Matrix(X,Y,'CND-in',classifier)
    ResultAUC_Dic.update({'CN-in':ListAuc})

    X=data1.iloc[:,[0,1,6]]
    ListAuc=Confucion_Matrix(X,Y,'CND-out',classifier)
    ResultAUC_Dic.update({'CN-out':ListAuc})

    X=data1.iloc[:,[0,1,5]]
    ListAuc=Confucion_Matrix(X,Y,'CN-inout',classifier)
    ResultAUC_Dic.update({'CN-inout':ListAuc})
    
    X=data1.iloc[:,[0,1,4]]
    ListAuc=Confucion_Matrix(X,Y,'AA-Out',classifier)
    ResultAUC_Dic.update({'AA-Out':ListAuc})
    
    X=data1.iloc[:,[0,1,3]]
    ListAuc=Confucion_Matrix(X,Y,'AA-in',classifier)
    ResultAUC_Dic.update({'AA-in':ListAuc})

    X=data1.iloc[:,[0,1,2]]
    ListAuc=Confucion_Matrix(X,Y,'AA-in-out',classifier)
    ResultAUC_Dic.update({'AA-in-Out':ListAuc})
    
    X=data1.iloc[:,[0,1,14]]
    ListAuc=Confucion_Matrix(X,Y,'Triadic',classifier)
    ResultAUC_Dic.update({'Triadic':ListAuc})

    X=data1.iloc[:,[0,1,15]]
    ListAuc=Confucion_Matrix(X,Y,'N2V',classifier)
    ResultAUC_Dic.update({'N2V':ListAuc})
    
    dictAUC.update({'LinearDiscriminantAnalysis':ResultAUC_Dic})
    print('____________________________','LinearDiscriminantAnalysis','____________________________________')
    print('____________________________','RandomForestClassifier','_________________________________________')
    classifier = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    
    ResultAUC_Dic={}
    roc_com_Measure(data1,classifier,'RandomForest',Str)
    X=data1.iloc[:,[0,1,13]]
    ListAuc=Confucion_Matrix(X,Y,'CNAH',classifier)
    ResultAUC_Dic.update({'CNAH':ListAuc})
    
    X=data1.iloc[:,[0,1,12]]
    ListAuc=Confucion_Matrix(X,Y,'CNHA',classifier)
    ResultAUC_Dic.update({'CNHA':ListAuc})
    
    X=data1.iloc[:,[0,1,11]]
    ListAuc=Confucion_Matrix(X,Y,'CNHA-AH',classifier)
    ResultAUC_Dic.update({'CNHA-AH':ListAuc})
    
    X=data1.iloc[:,[0,1,10]]
    ListAuc=Confucion_Matrix(X,Y,'RA-Out',classifier)
    ResultAUC_Dic.update({'RA-Out':ListAuc})
    
    X=data1.iloc[:,[0,1,9]]
    ListAuc=Confucion_Matrix(X,Y,'RA-in',classifier)
    ResultAUC_Dic.update({'RA-in':ListAuc})
    
    X=data1.iloc[:,[0,1,8]]
    ListAuc=Confucion_Matrix(X,Y,'RA-in-out',classifier)
    ResultAUC_Dic.update({'RA-in-Out':ListAuc})

    X=data1.iloc[:,[0,1,7]]
    ListAuc=Confucion_Matrix(X,Y,'CND-in',classifier)
    ResultAUC_Dic.update({'CN-in':ListAuc})

    X=data1.iloc[:,[0,1,6]]
    ListAuc=Confucion_Matrix(X,Y,'CND-out',classifier)
    ResultAUC_Dic.update({'CN-out':ListAuc})

    X=data1.iloc[:,[0,1,5]]
    ListAuc=Confucion_Matrix(X,Y,'CN-inout',classifier)
    ResultAUC_Dic.update({'CN-inout':ListAuc})
    
    X=data1.iloc[:,[0,1,4]]
    ListAuc=Confucion_Matrix(X,Y,'AA-Out',classifier)
    ResultAUC_Dic.update({'AA-Out':ListAuc})
    
    X=data1.iloc[:,[0,1,3]]
    ListAuc=Confucion_Matrix(X,Y,'AA-in',classifier)
    ResultAUC_Dic.update({'AA-in':ListAuc})

    X=data1.iloc[:,[0,1,2]]
    ListAuc=Confucion_Matrix(X,Y,'AA-in-out',classifier)
    ResultAUC_Dic.update({'AA-in-Out':ListAuc})
    
    X=data1.iloc[:,[0,1,14]]
    ListAuc=Confucion_Matrix(X,Y,'Triadic',classifier)
    ResultAUC_Dic.update({'Triadic':ListAuc})

    X=data1.iloc[:,[0,1,15]]
    ListAuc=Confucion_Matrix(X,Y,'N2V',classifier)
    ResultAUC_Dic.update({'N2V':ListAuc})
    
    dictAUC.update({'RandomForestClassifier':ResultAUC_Dic})
    print('____________________________','RandomForestClassifier','_________________________________________')
    print('____________________________','DecisionTreeClassifier','_________________________________________')
    classifier =DecisionTreeClassifier()
    
    ResultAUC_Dic={}
    
    X=data1.iloc[:,[0,1,13]]
    ListAuc=Confucion_Matrix(X,Y,'CNAH',classifier)
    ResultAUC_Dic.update({'CNAH':ListAuc})
    
    X=data1.iloc[:,[0,1,12]]
    ListAuc=Confucion_Matrix(X,Y,'CNHA',classifier)
    ResultAUC_Dic.update({'CNHA':ListAuc})
    
    X=data1.iloc[:,[0,1,11]]
    ListAuc=Confucion_Matrix(X,Y,'CNHA-AH',classifier)
    ResultAUC_Dic.update({'CNHA-AH':ListAuc})
    
    X=data1.iloc[:,[0,1,10]]
    ListAuc=Confucion_Matrix(X,Y,'RA-Out',classifier)
    ResultAUC_Dic.update({'RA-Out':ListAuc})
    
    X=data1.iloc[:,[0,1,9]]
    ListAuc=Confucion_Matrix(X,Y,'RA-in',classifier)
    ResultAUC_Dic.update({'RA-in':ListAuc})
    
    X=data1.iloc[:,[0,1,8]]
    ListAuc=Confucion_Matrix(X,Y,'RA-in-out',classifier)
    ResultAUC_Dic.update({'RA-in-Out':ListAuc})

    X=data1.iloc[:,[0,1,7]]
    ListAuc=Confucion_Matrix(X,Y,'CND-in',classifier)
    ResultAUC_Dic.update({'CN-in':ListAuc})

    X=data1.iloc[:,[0,1,6]]
    ListAuc=Confucion_Matrix(X,Y,'CND-out',classifier)
    ResultAUC_Dic.update({'CN-out':ListAuc})

    X=data1.iloc[:,[0,1,5]]
    ListAuc=Confucion_Matrix(X,Y,'CN-inout',classifier)
    ResultAUC_Dic.update({'CN-inout':ListAuc})
    
    X=data1.iloc[:,[0,1,4]]
    ListAuc=Confucion_Matrix(X,Y,'AA-Out',classifier)
    ResultAUC_Dic.update({'AA-Out':ListAuc})
    
    X=data1.iloc[:,[0,1,3]]
    ListAuc=Confucion_Matrix(X,Y,'AA-in',classifier)
    ResultAUC_Dic.update({'AA-in':ListAuc})

    X=data1.iloc[:,[0,1,2]]
    ListAuc=Confucion_Matrix(X,Y,'AA-in-out',classifier)
    ResultAUC_Dic.update({'AA-in-Out':ListAuc})
    
    X=data1.iloc[:,[0,1,14]]
    ListAuc=Confucion_Matrix(X,Y,'Triadic',classifier)
    ResultAUC_Dic.update({'Triadic':ListAuc})

    X=data1.iloc[:,[0,1,15]]
    ListAuc=Confucion_Matrix(X,Y,'N2V',classifier)
    ResultAUC_Dic.update({'N2V':ListAuc})
    
    dictAUC.update({'DecisionTreeClassifier':ResultAUC_Dic})
    #dict_Dataset_AUC={}
    #dict_Dataset_AUC.update({Str:dictAUC})
    print(dictAUC)
    d=pd.DataFrame(dictAUC)
    print(d)
    d.to_excel("result/AUC_COM"+Str+".xlsx", sheet_name='MLExcel')
    print('____________________________','DecisionTreeClassifier','_________________________________________')
    print('###########################################',Str,'##############################################')

    




        

