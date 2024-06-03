# Metrics

模型評估指標可以反應模型的一部分性能，其中不同任務的評估指標也都不盡相同，選擇合理的評估指標才能更準確的評價模型性能。各任務與相對的指標如下所示:

- Regression Metrics
  - MSE
  - MAE

- [Classification Metrics](classification.md)
  - Confuion Matrix
  - Accuracy, Precision, and Recall
  - F1-score
  - PR curve and ROC curve
    - FPR and TPR
  - AUC

- [Object Detection Metrics](object_detection.md)
  - Confuion Matrix
  - Precision and Recall
  - F1-score
  - IoU
  - PR curve
  - Average Precision (AP)
  - Mean Average Precision (mAP)

- [Segmentation Metrics](segmentation.md)
  - Confuion Matrix
  - Accuracy, Precision, and Recall
    - Pixel Accuracy (PA)
    - Class Pixel Accuracy (CPA)
    - Mean Pixel Accuracy (MPA)
  - IoU, MIoU
  - Dice coefficient
