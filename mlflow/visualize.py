import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(cm, trainset,save_path="confusion_matrix.png"):
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=trainset.classes)
    plt.figure(figsize=(12, 15))
    cm_display.plot(cmap=plt.cm.Blues, values_format='d')
    plt.xticks(rotation=45, ha='right')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print(f"[+] Confusion matrix saved to {save_path}")