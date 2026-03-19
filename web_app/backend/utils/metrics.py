from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

CLASS_NAMES = ['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']

def compute_overall_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {'accuracy': 0, 'f1_weighted': 0, 'per_class': {}}

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    per_class = {}
    for i in range(4):
        true_pos = cm[i, i]
        total_true = np.sum(cm[i, :])
        total_pred = np.sum(cm[:, i])
        
        precision = true_pos / total_pred if total_pred > 0 else 0
        recall = true_pos / total_true if total_true > 0 else 0
        iou = true_pos / (total_true + total_pred - true_pos) if (total_true + total_pred - true_pos) > 0 else 0
        
        per_class[CLASS_NAMES[i]] = {
            'precision': float(precision),
            'recall': float(recall),
            'iou': float(iou),
            'support': int(total_true)
        }
        
    return {
        'accuracy': float(acc),
        'f1_weighted': float(f1),
        'per_class': per_class
    }
