"""
Created on Tue Apr 18 2017

Predicting the Popularity of Online News 
dataset from UCI Machine Learning Repository
Roc graph

adopted the sklearn library http://scikit-learn.org/stable/

@author: juliana
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

csv_filename = "OnlineNewsPopularity.csv"
data = pd.read_csv(csv_filename)

# attributes list
title = data["n_tokens_title"]
content = data["n_tokens_content"]
unique_tokens = data["n_unique_tokens"]
non_stop_words = data["n_non_stop_words"]
non_stop_unique_tokens = data["n_non_stop_unique_tokens"]
hrefs = data["num_hrefs"]
self_hrefs = data["num_self_hrefs"]
imgs = data["num_imgs"]
videos = data["num_videos"]
average_token_length = data["average_token_length"]
num_keywords = data["num_keywords"]
c_lifestyle = data["data_channel_is_lifestyle"]
c_entertainment = data["data_channel_is_entertainment"]
c_bus = data["data_channel_is_bus"]
c_socmed = data["data_channel_is_socmed"]
c_tech = data["data_channel_is_tech"]
c_world = data["data_channel_is_world"]
kw_min = data["kw_min_min"]
kw_max_min = data["kw_max_min"]
kw_avg_min = data["kw_avg_min"]
kw_min_max = data["kw_min_max"]
kw_max_max = data["kw_max_max"]
kw_avg_max = data["kw_avg_max"]
kw_min_avg = data["kw_min_avg"]
kw_max_avg = data["kw_max_avg"]
kw_avg_avg = data["kw_avg_avg"]
min_shares = data["self_reference_min_shares"]
max_shares = data["self_reference_max_shares"]
avg_sharess = data["self_reference_avg_sharess"]
monday = data["weekday_is_monday"]
tuesday = data["weekday_is_tuesday"]
wednesday = data["weekday_is_wednesday"]
thursday = data["weekday_is_thursday"]
friday = data["weekday_is_friday"]
saturday = data["weekday_is_saturday"]
sunday = data["weekday_is_sunday"]
lDA_00 = data["LDA_00"]
lDA_01 = data["LDA_01"]
lDA_02 = data["LDA_02"]
lDA_03 = data["LDA_03"]
lDA_04 = data["LDA_04"]
global_subjectivity = data["global_subjectivity"]
global_sentiment_polarity = data["global_sentiment_polarity"]
global_rate_positive_words = data["global_rate_positive_words"]
global_rate_negative_words = data["global_rate_negative_words"]
rate_positive_words = data["rate_positive_words"]
rate_negative_words = data["rate_negative_words"]
avg_positive_polarity = data["avg_positive_polarity"]
min_positive_polarity = data["min_positive_polarity"]
max_positive_polarity = data["max_positive_polarity"]
avg_negative_polarity = data["avg_negative_polarity"]
min_negative_polarity = data["min_negative_polarity"]
max_negative_polarity = data["max_negative_polarity"]
subjectivity = data["title_subjectivity"]
sentiment_polarity = data["title_sentiment_polarity"]
subjectivity = data["abs_title_subjectivity"]
weekend = data["is_weekend"]

# target value
popular = []
data.share = data[" shares"]

for i in data.share:
    if i > 1400:
        popular.append(1)
    else:
        popular.append(0)

# all attributes
DataSet = list(
    zip(
        title,
        content,
        unique_tokens,
        non_stop_words,
        non_stop_unique_tokens,
        hrefs,
        self_hrefs,
        imgs,
        videos,
        average_token_length,
        num_keywords,
        c_lifestyle,
        c_entertainment,
        c_bus,
        c_socmed,
        c_tech,
        c_world,
        kw_min,
        kw_max_min,
        kw_avg_min,
        kw_min_max,
        kw_max_max,
        kw_avg_avg,
        min_shares,
        max_shares,
        avg_sharess,
        monday,
        tuesday,
        wednesday,
        thursday,
        friday,
        saturday,
        sunday,
        lDA_00,
        lDA_01,
        lDA_02,
        lDA_03,
        lDA_04,
        global_subjectivity,
        global_sentiment_polarity,
        global_rate_positive_words,
        global_rate_negative_words,
        rate_positive_words,
        rate_negative_words,
        avg_positive_polarity,
        min_positive_polarity,
        max_positive_polarity,
        avg_negative_polarity,
        min_negative_polarity,
        avg_negative_polarity,
        max_negative_polarity,
        subjectivity,
        sentiment_polarity,
        subjectivity,
        weekend,
    )
)
X = DataSet
y = popular

# Binarize the output
y = label_binarize(popular, classes=[0, 1])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=400)
classifier.fit(X_train, y_train)
fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])
roc_aucRF = auc(fpr, tpr)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
fNN, tNN, _ = roc_curve(y_test, knn.predict_proba(X_test)[:, 1])
roc_aucKNN = auc(fNN, tNN)

# NB
gnb = BernoulliNB()
gnb.fit(X_train, y_train)
fgnb, tgnb, _ = roc_curve(y_test, gnb.predict_proba(X_test)[:, 1])
roc_aucgnb = auc(fgnb, tgnb)

plt.figure()
plt.plot(fpr, tpr, label="RF(area = %0.2f)" % roc_aucRF)
plt.plot(fNN, tNN, label="KNN (area = %0.2f)" % roc_aucKNN)
plt.plot(fgnb, tgnb, label="NB (area = %0.2f)" % roc_aucgnb)

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
