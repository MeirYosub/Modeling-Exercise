import math
import random
from sklearn.metrics import accuracy_score
from statistics import mean,pstdev

def cross_validation(X,y,kfold,clf) -> (float,float,float,float):
    folds_index=list(range(1,kfold+1))*math.ceil(X.shape[0]/kfold)
    folds_index=folds_index[0:X.shape[0]]
    random.shuffle(folds_index)
    
    acc_train_fold=[]
    acc_valid_fold=[]
    for k in range(1,kfold+1):
        x_train_fold = X.iloc[[idx for idx in range(len(folds_index)) if folds_index[idx] != k]]
        y_train_fold = y.iloc[[idx for idx in range(len(folds_index)) if folds_index[idx] != k]]
        x_valid_fold = X.iloc[[idx for idx in range(len(folds_index)) if folds_index[idx] == k]]
        y_valid_fold = y.iloc[[idx for idx in range(len(folds_index)) if folds_index[idx] == k]]
        
        clf.fit(x_train_fold, y_train_fold)
        
        y_train_pred=clf.predict(x_train_fold)
        acc_train_fold.append(accuracy_score(y_train_fold, y_train_pred))
            
        y_valid_pred = clf.predict(x_valid_fold)
        acc_valid_fold.append(accuracy_score(y_valid_fold, y_valid_pred))
        
    acc_train = round(mean(acc_train_fold),3)
    std_train = round(pstdev(acc_train_fold),3)
    acc_valid = round(mean(acc_valid_fold),3)
    std_valid = round(pstdev(acc_valid_fold),3)
        
    return (acc_train,std_train,acc_valid,std_valid)