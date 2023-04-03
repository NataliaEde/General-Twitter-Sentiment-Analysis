# Project4, Tweeter Sentiment Analysis using NLP, Machine Learning, and Deep Learning

<img width="667" alt="Screen Shot 2023-03-21 at 11 36 20 AM" src="https://user-images.githubusercontent.com/44559346/226664474-dc1d9e64-e0dd-4c07-8272-e10302de45ca.png">


# INTRODUCTION

Today, customers are increasingly voicing their opinions on social media, review sites, and other online platforms. Because of these opportunities, customer feedback reaches a wider audience, and businesses are affected accordingly. Companies are now more likely to consider customer feedback and improve their brand or product accordingly.

Companies use sentiment analysis to understand the opinions, attitudes, and emotions underlying customer feedback. This approach seeks not only to measure customer satisfaction but also to monitor brand reputation in order to identify potential issues and areas of improvement. Sentiment analysis can be useful for a variety of applications, such as monitoring satisfaction with a service or product, analyzing posts on social media sentiment, detecting fake news, and predicting stock prices based on public opinion.

# PURPOSE

Our goal is to build a machine-learning model to predict if a given tweet is positive or negative.


# DATA AND METHODOLOGY

Our data was taken from Kaggle (https://www.kaggle.com/datasets/kazanova/sentiment140). It contains 1,600,000 tweets extracted using the Twitter API. First, we used the machine learning model to analyze 30,000 tweets. Then, we used the deep learning model for 1,000,000 tweets.

We focused on the following columns: Tweets and Target Variables (negative or positive).
target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
text: the text of the tweet (Lyx is cool)

Wordcloud can be used to create a visual representation of the most frequently used words and phrases in a given text. We highlighted the keywords used in positive and negative tweets.

![image](https://user-images.githubusercontent.com/44559346/226665783-4bfd792c-61ad-4427-8df7-5cdd6b7dd6b5.png)


In the positive wordclould tweets, "good", "lol", "nice", "hope" , "thank", and "love" were the most frequently used words. This might suggest that many of the positive tweets express positive sentiment, with a focus on feelings of hope, appreciation, and love. On the other hand, the words "want", "wish," "today", "work", "still", and "need," which are the most frequent negative tweets, possibly express frustration, dissatisfaction, and unfulfilled need.


## MACHINE LEARNING

First, we ran the 30,000 tweets through the following machine-learning models used for classification and regression tasks:
Naive Bayes is an algorithm that is based on Bayes' theorem, which describes the probability of an event based on prior knowledge of conditions that might be related to the event.
XGBoost (Extreme Gradient Boosting) is a powerful algorithm that uses gradient boosting to train decision trees.
Random Forest is an ensemble learning algorithm that constructs a set of decision trees and combines their predictions to make a final prediction.

For each model, we used three different vector-type techniques for text representation:
TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure that reflects the importance of a term in a document.
Vector Count represents each document as a vector of word frequencies.
Word2Vec is a neural network-based technique that learns continuous vector representations of words.

This method yielded the following results:

![image](https://user-images.githubusercontent.com/44559346/226665375-08fd317d-5362-488b-aaff-e62bd010ed41.png)


## DEEP LEARNING

We felt the machine learning results to be inadequate, so we implemented the LSTM (Long Short-Term Memory) model, a type of Recurrent Neural Network (RNN) that is used to process sequential data, learn complex patterns, and make predictions about future events. LSTM is capable of handling a larger data set, so we increased our data to 1 million tweets.

For the complete code, you can visit my github: [provide address]
RESULTS
ACCURACY: 0.7863839864730835
LOSS: 0.4514560103416443


Confusion Matrix

![image](https://user-images.githubusercontent.com/44559346/226665547-69bd0645-5cc1-4957-aeb9-fe756add56c1.png)

Training and Validation Accuracy

We plotted the accuracy and loss values over the epochs of the model training in order to how the model performs in the neural network setting.
As the training accuracy increases, so does the validation accuracy. The same occurs with the training loss graph: loss decreases, and so does validation loss. This suggests that the model learns the patterns in the data and adapts to new data. In both graphs, when the epochs pass 6, the training and the validation approach one another. With higher than 10 epochs, the model will overfit the training data and not make accurate predictions.

![image](https://user-images.githubusercontent.com/44559346/226665904-7e192a6c-3dd2-4e7e-9e4a-a5f5bd277188.png)


![image](https://user-images.githubusercontent.com/44559346/226665932-8a439e97-796a-4272-86b3-05068dac2190.png)



## CONCLUSION

In conclusion, we employed various models for sentiment analysis and found that the LSTM model outperformed the others. With training on one million tweets, the model achieved a prediction accuracy of 79%. To demonstrate its real-world application, we used the model to analyze airline feedback tweets, accurately predicting sentiments 71% of the time. Further analysis will be useful to determine the model's limitations and areas for improvement.

To gain further insights(see in LTSM Notebook), we applied an LDA model for topic modeling, focusing specifically on negative tweets. The topics that emerged revealed that customers are predominantly dissatisfied with US Airlines' customer service and frequent delays. This valuable feedback can help airlines address these issues and make improvements in the areas that matter most to their customers. By continuously refining and applying such sentiment analysis models, businesses can gain actionable insights from customer feedback and enhance their products and services accordingly.
