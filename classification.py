####
This script assumes that there is a file named 'data.pt' that contains the features of the sounds. 

####

import torch
import numpy as np

# ML libraries 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import preprocessing 
 from sklearn.svm import SVC

 
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)


x=torch.load('data.pt')
y=np.load('label.npy')


labeler = lambda ch: 'char' if len(ch) > 1 else 'letter'
y = [ labeler(i) for i in label]


label_encoder = preprocessing.LabelEncoder() 

Y= label_encoder.fit_transform(y) 


X_train, X_Val, y_train, y_val = train_test_split(d, Y, test_size=0.1,  stratify=y,      random_state=random_seed, shuffle=True)


# Create an SVM classifier
svm = SVC(random_state=42)

svm.fit(X_resembled, y_resembled)

y_pred = svm.predict(X_Val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)

print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))
print(f"Accuracy: {accuracy}")


