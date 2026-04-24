# Model Comparison: SVM vs. CNN (Live Camera Tests)

Based on your recent live camera testing sessions, here is the comparative breakdown of both models across the four key metrics. 

> [!NOTE]
> The SVM test was performed on **4** live samples, while the CNN TFLite test was performed on **5** live samples.

| Metric | SVM Model | CNN TFLite Model | Comparison |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 75.00% | 80.00% | CNN is **Higher** (+5.00%) |
| **Precision** | 0.8333 | 0.6667 | SVM is **Higher** (+0.1666) |
| **Recall** | 0.8333 | 0.7500 | SVM is **Higher** (+0.0833) |
| **F1-Score** | 0.7778 | 0.7000 | SVM is **Higher** (+0.0778) |

### Key Takeaways:
1. **Accuracy Edge**: The **CNN** performed slightly better in raw accuracy during the session, meaning it got a higher percentage of the total captures right.
2. **Quality of Predictions**: The **SVM** scored higher in Precision, Recall, and F1-Score. This implies that when the SVM makes a prediction, it is proportionally more reliable, and it has a slightly stronger balance between false positives and false negatives among the tested cases.

*(Remember to test on a larger set of live captures for a fully comprehensive benchmark between the two!)*
