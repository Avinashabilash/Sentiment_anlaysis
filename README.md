# Sentiment Analysis with BERT & Machine Learning
Sentiment analysis is the process of automatically determining whether a piece of text expresses a positive, negative, or neutral opinion. It is widely used in applications such as product reviews, customer feedback, and social media monitoring.

# what is ?
Sentiment Analysis with BERT & Machine Learning is a text classification project that predicts whether a review is Positive, Negative, or Neutral. It compares traditional models (Logistic Regression, Random Forest with TF-IDF) against a fine-tuned BERT transformer, showing how deep learning achieves higher accuracy in understanding sentiments.

# 🎯 Goals
Preprocess and clean customer review text data.
Build baseline models (Logistic Regression, Random Forest, etc.) using TF-IDF features.
Fine-tune BERT (Bidirectional Encoder Representations from Transformers) for sentiment classification.
Compare performance metrics (accuracy, precision, recall, F1-score) between traditional ML and BERT.
Understand customer opinions better to help businesses improve decision-making.
🛠 Technical Details
Data Preprocessing: Tokenization, stopword removal, lowercasing, lemmatization, and TF-IDF vectorization.
Baseline Models: Logistic Regression, Random Forest, and Naïve Bayes.
Deep Learning Model: BERT fine-tuned for text classification.
Evaluation Metrics: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
Libraries Used:
pandas, numpy, matplotlib, seaborn
scikit-learn for ML models
transformers for BERT fine-tuning
torch for deep learning
# 📊 Case Explanation
By analyzing customer reviews, businesses can:

Detect positive/negative trends in feedback.

Improve customer satisfaction strategies.

Enhance product quality and service based on insights.

<p align="center">
  <img src="images/accuracy.png" width="290" alt="Accuracy Screenshot">
</p>

<p align="center">
  <img src="images/result.png" width="290" alt="Result Screenshot">
</p>

<p align="center">
  <img src="images/workflow.png" width="250" height="320" alt="Workflow">
</p>


