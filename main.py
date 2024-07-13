from scripts.data_processing import read_datasets
from scripts.feature_engineering import extract_features
from scripts.model_training import train
from scripts.evaluation import evaluate
from scripts.visualization import plot_confusion_matrix, plot_roc_curve

if __name__ == "__main__":
    print("Reading datasets...")
    x, y = read_datasets()
    
    print("Extracting features...")
    x = extract_features(x)
    
    print("Training model...")
    y_test, y_pred = train(x, y)
    
    print("Evaluating model...")
    cm, cm_normalized, roc_auc = evaluate(y_test, y_pred)
    
    print("Plotting results...")
    plot_confusion_matrix(cm)
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    plot_roc_curve(y_test, y_pred)
