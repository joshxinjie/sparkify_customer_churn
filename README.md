# Sparkify Customer Churn Prediction with PySpark

Draft of the medium [post](https://medium.com/@joshxinjielee/customer-churn-prediction-with-pyspark-on-ibm-watson-studio-aws-and-databricks-de57a2ffb25b)

## Description
Predict customer churn with PySpark for an imaginary digital music service called Sparkify. Sparkify has a free-tier and a premium subscription plan. Users can upgrade, downgrade or cancel their service at any time. Churn means downgradding from premium to free tier or cancelling the service. If we can predict users who will churn, the company can offer them discounts and incentives to entice them to stay.

## File Description
1. Sparkify.ipynb: The original prototype of the customer churn model trained on the local machine with the small instance of the dataset. This code does not include further improvements made to the model when I trained them on the cloud services. For a more updated version of the code, check out the notebooks ran on the various cloud services.
2. Sparkify_AWS.ipynb: The customer churn model trained on the full dataset on Amazon's AWS EMR service. Contains additional improvements not implemented in Sparkify.ipynb. This code does not have any EDA as AWS EMR does not have any visualization libraries pre-installed. You will need to manually install the libraries if you wnt to use them.
3. Sparkify_IBM.ipynb: The customer churn model trained on the medium-sized instance of the data file on IBM Watson Studio. Also contains additional improvements not implemented in Sparkify.ipynb. This code contains exploratory data analysis performed on the medium-sized instance of the dataset.
4. sparkify_medium_databricks.ipynb: The customer churn model trained on the medium-sized instance of the data file on Databricks. This file was actually trained on the full platform of Databricks instead of the free Community Edition. If you want the run the code on the free community edition, change the file to the small instance. I tested that the free Community Edition will work with the small dataset, however it is likely to encounter memory issues when handling the medium-sized dataset.
5. Sparkify_full_feat_imp.ipynb: Plots the most important features of the Gradient Boosted Tree model trained on the full dataset.

## Dataset
The dataset is a .json file that keeps track of timestamped events of the following actions performed on the digital music service:
1. play a song
2. login
3. listening to an advertisement
4. downgrading subscription
5. cancelling subscription
6. ...

There are 3 different sizes of the dataset available:
1. mini_sparkify_event_data.json: the smallest instance of the dataset (125 mb)
2. medium-sparkify-event-data.json: a medium-sized instance of the dataset (237 mb)
3. sparkify_event_data.json: the full dataset (12 gb)

You can build a prototype of the model with the samller instances of the dataset, then train your final model with the full dataset.

Data download links
* Link to small-sized subset of Sparkify data (125 mb): https://drive.google.com/open?id=1FwuyO5apNwy8q6BpG-_EIqqN-ED0X9tx
* Link to medium-sized subset of Sparkify data (237 mb): https://drive.google.com/open?id=17Lys6v7LOcAWFMHXwXwslUWj04LoNWp1
* Link to full Sparkify dataset on AWS (12 gb): s3n://dsnd-sparkify/sparkify_event_data.json

## Instructions
1. Download the appropriate data file
2. If you are using a cloud service, follow the instructions in the blog post
3. If you are running the code on your local computer, place the data file in the dame directory as the jupyter notebook and run the notebook

## Motivation
The motivation to do this project is to lean customer churn predictive modeling, big data tools (PySpark) and cloud computing services (such as IBM Watson Studio, AWS and Databricks).

## Side Notes
#### 1) Errors you may encounter when running on the code on the cloud
When running the notebooks on any cloud service, you might encounter issues such as error messages stating that the session isn't active, the notebook crashing and restarting, etc. This likely meant that the cluster have encountered memory issues and the easiest wy to resolve it will be to terminate your current cluster, create a new cluster and connect your notebook to the new cluster. Afterwards, you can run whichever cells that have not been run.

When running the notebook on AWS EMR, you might encounter these two error/exception messages when running certain cells.
```
TypeError: object of type 'NoneType' has no len()
```
or
```
KeyError: 14933 (the number could be different)
```
Usually these code do not mean that your code is faulty. You can ignore these messages and your code will still finish executing. More details in the blog post.

#### 2) PySpark's MulticlassClassificationEvaluator returns weighted f1-score for a binary classification task, not the actual f1-score
When using PySpark's MulticlassClassificationEvaluator to evaluate the f1-score for a binary classification task, you will get a weighted f1-score that treats both labels 0 and 1 as seperate classes. This might lead to an overly optimistically high f1-score.

For example, consider that the model generated the following confusion matrix:

TN:53.0 | FP:4.0

FN:17.0 | TP: 2.0

Running
```
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print("Logistic Regression Model --- F1-Score is: ")
print(evaluator.evaluate(lr_results, {evaluator.metricName: "f1"}))
```
will produce
```
Logistic Regression Model --- F1-Score is: 
0.665984251968504
```
If you were to maunally calculate the weighted f1-score, treating both labels 0 and 1 as seperate classes, you will get
```
TN = 53.0
FP = 4.0
FN = 17.0
TP = 2.0

precision_1 = TP/(TP+FP)
recall_1 = TP/(TP+FN)
f1_1 = 2 * (precision_1 * recall_1)/(precision_1 + recall_1)
weight_1 = (TP + FN)/(TP+FP+FN+TP)

precision_2 = TN/(TN+FN)
recall_2 = TN/(TN+FP)
f1_2 = 2 * (precision_2 * recall_2)/(precision_2 + recall_2)
weight_2 = (TN + FP)/(TP+FP+FN+TP)

multi_f1 = (weight_1 * f1_1 + weight_2 * f1_2)/(weight_1 + weight_2)
print(multi_f1)

OUTPUT: 0.6659842519685039
```

Clearly, this isn't what we want. The actual f1-score should be

precision = $\\frac{TP}{TP+FP}$ = $\\frac{2}{2+4}$ = $0.3333$

recall = $\frac{TP}{TP + FN}$ = $\frac{2}{2+17}$ = $0.1053$

f1-score = $2 * \frac{precision * recall}{precision + recall}$ = $2 * \frac{0.3333 * 0.1053}{0.3333 + 0.1052}$ = 0.16

#### 3) Engineering moving averages of user statistics and differences between 1-month lagged values and the moving averages
In the code for Sparkify.ipynb, you will notice the implementation for generating moving averages of the same statistics. In addition, I have included codes for generating differences between 1-month lagged features and the moving averages. It is a good idea to create differences between the two, because most machine learning models (perhaps with the exception of neural networks) do not innately extract feature interactions. These moving averages and differences were generated with the expectation that there might be more than 2 months of data available on the full dataset.

However, this was not the case, and the 1-month lagged features of the second month were equivalent to the moving averages. Hence, I dropped the moving averages and differences features from the final model since they would bot be adding any new information.
I have decided to keep their implementation in the code, since they might become useful in future scenarios when we have longer periods of data available.

## Installations
Anaconda (Pandas, Numpy, MatPlotLib), PySpark, Seaborn
