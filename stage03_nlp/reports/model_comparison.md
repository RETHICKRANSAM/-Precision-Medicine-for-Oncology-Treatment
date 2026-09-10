# Model Comparison: Clinical Urgency Classification

Evaluation performed on identical test set. Primary ranking metric: **Macro F1**.

| Model        |   Accuracy |   Macro_Precision |   Macro_Recall |   Macro_F1 |   Weighted_F1 |   Balanced_Accuracy |
|:-------------|-----------:|------------------:|---------------:|-----------:|--------------:|--------------------:|
| BiLSTM       |     0.953  |            0.9539 |         0.9533 |     0.9534 |        0.9531 |              0.9533 |
| ClinicalBERT |     0.8565 |            0.8576 |         0.8638 |     0.8564 |        0.8533 |              0.8638 |
| SVM_TFIDF    |     0.7877 |            0.7864 |         0.7955 |     0.7869 |        0.7826 |              0.7955 |

## Per-Class Breakdown & High-Urgency Recall
| Model | Low F1 | Moderate F1 | High F1 | **High Recall** |
| :--- | :--- | :--- | :--- | :--- |
| **SVM_TFIDF** | 0.8503 | 0.6897 | 0.8208 | **0.8589** |
| **BiLSTM** | 0.9669 | 0.9414 | 0.9520 | **0.9597** |
| **ClinicalBERT** | 0.8897 | 0.7935 | 0.8859 | **0.9395** |