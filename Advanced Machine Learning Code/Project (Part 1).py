# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 23:09:32 2019

@author: yasha
"""

#STEP 1: Accessing the whole Enron Dataset and finding the top 10 email addresses
#Note: This might take a while to run. It took about an hour for me to run it. 

import os
from email.parser import Parser
from collections import Counter
rootdir="C:\\Users\\yasha\\enron_mail_20150507\\maildir\\"
to_email_list=[]
from_email_list=[]
email_body=[]
def email_analyse(inputfile, to_email_list, from_email_list, email_body): 
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
            
for directory, subdirectory, filenames in os.walk(rootdir):
	for filename in filenames:
		email_analyse(os.path.join(directory, filename), to_email_list, from_email_list, email_body)

print("\nTo email adresses: \n")
print(Counter(to_email_list).most_common(10))

print("\nFrom email adresses: \n")
print(Counter(from_email_list).most_common(10))