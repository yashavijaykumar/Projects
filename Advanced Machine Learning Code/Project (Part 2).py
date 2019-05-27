# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 23:09:32 2019

@author: yasha
"""

#STEP 1: Accessing Richard Shapiro's email folders

import os
import csv
from email.parser import Parser
import pandas as pd
rootdir="C:\\Users\\yasha\\enron_mail_20150507\\maildir\\shapiro-r\\"
to_email_list=[]
from_email_list=[]
subject_email_list=[]
email_body=[]
def email_analyse(inputfile, to_email_list, from_email_list, subject_email_list, email_body): 
    with open(inputfile, "r") as f:  
        data=f.read() 
        email = Parser().parsestr(data) 
        if email['to']:
            email_to = email['to']
            email_to = email_to.replace("\n", "")
            email_to = email_to.replace("\t", "")
            email_to = email_to.replace(" ", "")
            email_to = email_to.split(",")
            for email_to_1 in email_to:
                to_email_list.append(email_to_1)
                from_email_list.append(email['from'])
        if email['subject']:
            email_sub = email['subject']
            subject_email_list.append(email_sub)
            email_body.append(email.get_payload())
            
for directory, subdirectory, filenames in os.walk(rootdir):
	for filename in filenames:
		email_analyse(os.path.join(directory, filename), to_email_list, from_email_list, subject_email_list, email_body)
#to
with open("to_email_list.txt", "w") as f:
	for to_email in to_email_list:
		if to_email:
			f.write(to_email)
			f.write("\n")
#with open("to_email_list.txt", "r") as f:
with open("to_email.csv", "w") as f_out:
    a = csv.writer(f_out)
    for i in to_email_list:
        a.writerow([i])

#from
with open("from_email_list.txt", "w") as f:
	for from_email in from_email_list:
		if from_email:
			f.write(from_email)
			f.write("\n")
#with open("from_email_list.txt", "r") as f:
with open("from_email.csv", "w") as f_out:
        a = csv.writer(f_out)
        for i in from_email_list:
        	a.writerow([i])
#subject
with open("subject_email_list.txt", "w") as f:
	for subject_email in subject_email_list:
		if subject_email:
			f.write(subject_email)
			f.write("\n")
#with open("subject_email_list.txt", "r") as f:
with open("subject_email.csv", "w") as f_out:
        a = csv.writer(f_out)
        for i in subject_email_list:
        	a.writerow([i])
#body
with open("email_body.txt", "w") as f:
	for email_bod in email_body:
		if email_bod:
			f.write(email_bod)
			f.write("\n")

#Putting the email bodies in a Dataframe 
info1=pd.DataFrame(columns=['Body'])
for i in email_body:
    	info1=info1.append({'Body':[i]},ignore_index=True)
writer = pd.ExcelWriter('ExcelBody.xlsx', engine='xlsxwriter')
info1.to_excel(writer, sheet_name='Sheet1')
writer.close()
            
#STEP 2: Data processing - Clean the email bodies
import pandas as pd
import re
from nltk.corpus import stopwords
import csv

def removeSpecialCharacter(data):
    return re.sub('[^a-zA-Z\n\.]', ' ', data)
    
dataframe = pd.read_excel("ExcelBody.xlsx", sheet_name=0)
dataframe['Body'] = dataframe['Body'].str.replace(r'\\n', '\n', regex=True)
dataframe['Body'] = dataframe['Body'].str.replace(r'\b\w{1,3}\b', '')
dataframe['Body'] = dataframe['Body'].apply(removeSpecialCharacter)
writer = pd.ExcelWriter('ExcelBodyCleaned.xls', engine='xlsxwriter')
dataframe.to_excel(writer, sheet_name='Sheet1')
writer.close()

stop = stopwords.words('english')
DataFrame = pd.read_excel("ExcelBodyCleaned.xls", sheet_name=0)
DataFrame['Body'].apply(lambda x: [item for item in x if item not in stop])
writer = pd.ExcelWriter('ExcelBodyCleanedstop.xls', engine='xlsxwriter')
DataFrame.to_excel(writer, sheet_name='Sheet1')
writer.close()

import pandas as pd
data_xls = pd.read_excel('ExcelBodyCleanedstop.xls', 'Sheet1', index_col=None)
data_xls.to_csv('ExcelBodyCleanedstop.csv', encoding='utf-8')

with open("ExcelBodyCleanedstop.csv", "r") as fp_in, open("ExcelBodyCleanedstop1.csv", "w") as fp_out:
    reader = csv.reader(fp_in)
    writer = csv.writer(fp_out)
    for row in reader:
        del row[0]
        writer.writerow(row)

with open("ExcelBodyCleanedstop1.csv",'r') as f, open("output.csv",'w') as f1:
   next(f) # skip header line
   for line in f:
        f1.write(line)
        
#STEP 3: Converting text to vectors
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

colnames=['Body'] 
email_body = pd.read_csv('output.csv', names=colnames, header=None)
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
stopwords = ENGLISH_STOP_WORDS.union(['ect', 'hou', 'com', 'recipient', 'your', 'will', 'their', 'http', 'know', 'thanks', 'said', 'html', 'attached'])
v=TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.5, min_df=2)
x=v.fit_transform(email_body['Body'])
email_body['BodyVect']=list(x)

#STEP 4: Apply clustering methods K-means, Mini batch k-means
from sklearn.cluster import KMeans
from time import time
n_clusters = 3
clf = KMeans(n_clusters=n_clusters, 
            max_iter=100, 
            init='k-means++', 
            n_init=1)
t0 = time()
labels = clf.fit_predict(x)
print("K-means done in %0.3fs" % (time() - t0))
print()

# Let's plot this with matplotlib to visualize it.
# First we need to make 2D coordinates from the sparse matrix. 
from sklearn.decomposition import PCA
X_dense = x.todense()
pca = PCA(n_components=2).fit(X_dense)
coords = pca.transform(X_dense)

#For K-means clustering:
# This array needs to be at least the length of the n_clusters.
label_colors = ["#2AB0E9", "#2BAF74", "#D7665E", "#CCCCCC", 
                "#D2CA0D", "#522A64", "#A3DB05", "#FC6514"]
colors = [label_colors[i] for i in labels]

plt.scatter(coords[:, 0], coords[:, 1], c=colors)
#Plot the cluster centers
centroids = clf.cluster_centers_
centroid_coords = pca.transform(centroids)
plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=200, linewidths=2, c='#444d60')
print ("KMeans clustering \n")
plt.show()

#MINI BATCH KMEANS 
# For larger datasets use mini-batch KMeans, so we dont have to read all data into memory.
from sklearn.cluster import MiniBatchKMeans
batch_size = 500
clf_mini = MiniBatchKMeans(n_clusters = 3, batch_size=batch_size, max_iter=100)  
t1 = time()
labels_mini = clf_mini.fit_predict(x)
print("mini-batch k-means done in %0.3fs" % (time() - t1))
print()

#Similarly for mini batch k-means:
# Lets plot it again, but this time we add some color to it.
# This array needs to be at least the length of the n_clusters.
label_colors = ["#2AB0E9", "#2BAF74", "#D7665E", "#CCCCCC", 
                "#D2CA0D", "#522A64", "#A3DB05", "#FC6514"]
colors = [label_colors[i] for i in labels]

plt.scatter(coords[:, 0], coords[:, 1], c=colors)
#Plot the cluster centers
centroids = clf_mini.cluster_centers_
centroid_coords = pca.transform(centroids)
plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=200, linewidths=2, c='#444d60')
print ("Mini-Batch KMeans clustering \n")
plt.show()

#STEP 5: GET FEATURE NAMES - top 10 features for one email,
#                            top 10 features for all email bodies
#                            and top features per cluster
features = v.get_feature_names()

def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df

def top_feats_in_doc(x, features, row_id, top_n=25):
    row = np.squeeze(x[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(x, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    if grp_ids:
        D = x[grp_ids].toarray()
    else:
        D = x.toarray()
    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

print ("The top 10 features in one email body are: \n" )
print (top_feats_in_doc(x, features, 1, 10))

print ("The top 10 features in all the email bodies are: \n" )
print (top_mean_feats(x, features, None, 0.1, 10))

def top_feats_per_cluster(x, y, features, min_tfidf=0.1, top_n=25):
    dfs = []

    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label) 
        feats_df = top_mean_feats(x, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    X = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("cluster = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(X, df.score, align='center', color='#7530FF')
        ax.set_yticks(X)
        ax.set_ylim([-1, X[-1]+1])
        yticks = ax.set_yticklabels(df.features)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()

#Use this to print the top terms per cluster with matplotlib.
print ("KMeans clustering top features per cluster \n")
plot_tfidf_classfeats_h(top_feats_per_cluster(x, labels, features, 0.1, 25))

#Use this to print the top terms per cluster with matplotlib.
print ("Mini-Batch KMeans clustering top features per cluster \n")
plot_tfidf_classfeats_h(top_feats_per_cluster(x, labels_mini, features, 0.1, 25))

#STEP 6: Finding emails linked to top terms. 
#        the query here, for now, is 'California'. It can be changed by the user.
from sklearn.metrics.pairwise import linear_kernel

def query(self, keyword, limit):
  vec_keyword = self.vec.transform([keyword])
  cosine_sim = linear_kernel(vec_keyword, self.vec_train).flatten()
  related_email_indices = cosine_sim.argsort()[:-limit:-1]
  print(related_email_indices)
  return related_email_indices

def find_email_by_index(self, i):
  return self.emails.as_matrix()[i]

cosine_sim = linear_kernel(x[0:1], x).flatten()

query = "california"

# Transform the query into the original vector
vec_query = v.transform([query])

cosine_sim = linear_kernel(vec_query, x).flatten()

# Find top 10 most related emails to the query.
related_email_indices = cosine_sim.argsort()[:-10:-1]
# print out the indices of the 10 most related emails.
print(related_email_indices)

first_email_index = related_email_indices[0]
mat_fin = info1.as_matrix()[first_email_index]
df_query = pd.DataFrame(mat_fin)
print("Emails related to query '",query,"' are saved in a text file")
df_query.to_csv(r'query_emailbodies.txt', header=None, index=None, sep=' ', mode='a')