import os
import csv
docdir = 'C:\\Users\\Lurye\\Desktop\\Research\\HR'
if os.getcwd()!=docdir:
    os.chdir(docdir)

with open("Kaspersky - Glassdoor Reviews - 2019-08-22.csv") as rf1:
    reader = csv.reader(rf1, dialect='excel', delimiter=',')
    data = []
    for row in reader:
        #print(row)
        #I just checked that it parsed csv right. It did, so omitting this line.
        if "Employer Name" in row:
            header = row
        elif 'AO Kaspersky Lab' in row:
            data.append(row)
    rf1.close()

header_dict = dict(zip(header, range(len(header))))
#this is for convenience purpose - it's easier to work with field names that numbers, but can be exsessive

def getkey(data, key):
    d = {}
    for item in data:
        temp = item[header_dict[key]]
        if temp in d.keys():
            d[temp] += 1
        else:
            d.update({temp: 1})
    return d
countries = getkey(data, 'Review Country Name')

with open("countries.csv", mode='w', newline='') as wf1:
    writer = csv.writer(wf1, dialect='excel', delimiter=';')
    writer.writerow(['Country', 'Number of reviews'])
    for key in countries.keys():
        writer.writerow([key, countries[key]])
    wf1.close()

#jobtitles = getkey(data, 'Job Title')
#commented because contains no meaningful data

def GetText():
    corpus = []
    k = header_dict['Headline']
    for item in data:
        corpus.append(item[k].lower().strip() + " ")
    return corpus

def GetScore():
    Score = []
    k = header_dict["Overall Satisfaction"]
    for item in data:
        Score.append(item[k])
    return Score

def GetScorePN():
    Score = []
    k = header_dict["Overall Satisfaction"]
    for item in data:
        if float(item[k]) > 3.0:
            Score.append('Positive')
        else:
            Score.append('Negative')
    return Score

from sklearn.feature_extraction.text import CountVectorizer
import nltk
#in case you need to download extra corpii, mind the proxy settings
#nltk.download('punkt')
#nltk.download('stopwords')
#from nltk.corpus import stopwords

t = GetText()
vectorizer = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize, stop_words='english')
word_count = vectorizer.fit_transform(t)    #X
from sklearn.feature_extraction.text import TfidfTransformer
#-------------------------------skip this and go straight to the train/test split--------------------
#xfmer = TfidfTransformer()
#word_tfidf = xfmer.fit_transform(word_count)    #X_

#scores = GetScore()
scores = GetScorePN()
from sklearn.model_selection import train_test_split
docs_train, docs_test, y_train, y_test = train_test_split(t, scores, test_size = 0.20, random_state = 12)
Vzer= CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, stop_words='english')
docs_train_counts = Vzer.fit_transform(docs_train)
Tfmer = TfidfTransformer()
docs_train_tfidf = Tfmer.fit_transform(docs_train_counts)
docs_test_counts = Vzer.transform(docs_test)
docs_test_tfidf = Tfmer.transform(docs_test_counts)


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(docs_train_tfidf, y_train)
y_pred = clf.predict(docs_test_tfidf)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
accuracy_score(y_test, y_pred)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#sanity check - let's look at the top features
#a function that fetches the feature name from vectorizer using the classifier coefficients
feature_to_coef = {
    word: coef for word, coef in zip(
        Vzer.get_feature_names(), clf.coef_[0]
    )
}
#Now let's print those features from the top of the list
for best_positive in sorted(
    feature_to_coef.items(),
    key=lambda x: x[1],
    reverse=True)[:5]:
    print (best_positive)
#And from the bottom of the list
for best_negative in sorted(
    feature_to_coef.items(),
    key=lambda x: x[1])[:5]:
    print (best_negative)


import datetime
def timespan(data):
    min = datetime.datetime.strptime(data[0][header_dict["Review Date"]], '%Y-%m-%d')
    #datetime.strptime method requires to specify string format with '%Y-%m-%d %H:%M:%S' being the most robust use case
    max = min
    for item in data:
        review_date = datetime.datetime.strptime(item[header_dict["Review Date"]], '%Y-%m-%d')
        if review_date < min:
            min = review_date
        elif review_date > max:
            max = review_date
    return min, max


#emplen = getkey(data, 'Length of Employment')
#SL = getkey(data, 'Senior Leadership')

import matplotlib.pyplot as plt
#------------------plotting overview------------------------
fig, axes = plt.subplots(3, 2)
axes[0, 0].set_title("Number of reviews per country")
emplen = sorted(getkey(data, 'Length of Employment').items(), reverse = True, key=lambda x: x[1])
x00 = [i[0] for i in emplen]
y00 = [int(i[1]) for i in emplen]
axes[0,0].bar(x00, y00)
axes[0,0].set_title("Length of employment")
satisfaction = sorted(getkey(data, 'Overall Satisfaction').items(), reverse = True, key=lambda x: x[1])
x01 = [i[0] for i in satisfaction]
y01 = [int(i[1]) for i in satisfaction]
axes[0, 1].bar(x01, y01)
axes[0, 1].set_title("Satisfaction", pad=-40)
opp = sorted(getkey(data, 'Career Opportunities').items(), reverse = True, key=lambda x: x[1])
x10 = [i[0] for i in opp]
y10 = [int(i[1]) for i in opp]
axes[1, 0].bar(x10, y10)
axes[1, 0].set_title("Career Opportunities")
CB = sorted(getkey(data, 'Compensation & Benefits').items(), reverse = True, key=lambda x: x[1])
x11 = [i[0] for i in CB]
y11 = [int(i[1]) for i in CB]
axes[1,1].bar(x11, y11)
axes[1,1].set_title("Compensation & Benefits")
recommend = getkey(data, 'Recommend to Friend')     #categorical
x20 = [i[0] for i in recommend.items()]
y20 = [int(i[1]) for i in recommend.items()]
axes[2, 0].bar(x20, y20)
axes[2, 0].set_title("Recommend to Friend")
BO = getkey(data, 'Business Outlook')   #categorical
x21 = [i[0] for i in BO.items()]
y21 = [int(i[1]) for i in BO.items()]
axes[2, 1].bar(x21, y21)
axes[2, 1].set_title("Business Outlook")
fig.suptitle("Analysis of 346 reviews on Glassdoor from 2009 to 2019")
fig.tight_layout()
plt.show()

#---------------plotting countries--------
countries_sorted = sorted(getkey(data, 'Review Country Name').items(), reverse = True, key=lambda x: x[1])
#using lambda function to sort over dictionary values
x00 = [i for i in range(len(countries_sorted))]
x00_labels = [i[0] for i in countries_sorted]
y00 = [int(i[1]) for i in countries_sorted]
plt.bar(x00, y00, align="center")
plt.xticks(x00, x00_labels)
plt.tick_params(labelrotation=35, size=11)
#.set_xticklabels(labels, rotation=45)
plt.title("Number of reviews per country")
plt.tight_layout()
plt.show()

#finally we get to a corpus of words
bulk_text = ''
for item in data:
    bulk_text += item[header_dict['Headline']] + ' '





from sklearn.feature_extraction.text import CountVectorizer

