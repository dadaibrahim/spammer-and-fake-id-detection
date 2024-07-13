import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def evaluate(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print('Confusion matrix, without normalization')
    print(cm)
    
    print('Normalized confusion matrix')
    print(cm_normalized)
    
    print('Classification report')
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Genuine']))
    
    print("ROC AUC Score:", roc_auc)
    
    return cm, cm_normalized, roc_auc
