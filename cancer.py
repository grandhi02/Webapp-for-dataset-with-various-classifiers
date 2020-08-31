import matplotlib.pyplot as plt
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score,f1_score
import numpy as np
import warnings   # it is important to remove warnings
warnings.filterwarnings('ignore')


st.title(""" 
Explore different classifier

""")

##  we get selectbox by below code
#dataset_name=st.selectbox("Select datset",("iris","cancer","winedataset"))

#st.write("dataset selected is "+dataset_name +" dataset" )


# to get sidebar we use the below code.
dataset=st.sidebar.selectbox("Select dataset",("iris","cancer","winedataset"))

st.write("Dataset selected is "+dataset +" dataset" )

classifier=st.selectbox("select classifier",("knn","randomforestclassifier","svm"))
st.write("classfier selected is "+classifier)

def get_dataset(dataset):
    if dataset=="iris":
        data=datasets.load_iris()
    elif dataset=="cancer":
        data=datasets.load_breast_cancer()
    else:
        data=datasets.load_wine()
    x=data.data
    y=data.target
    return x,y
X,y=get_dataset(dataset)
st.write(""" shape of input """ ,X.shape)
st.write(""" No of unique targets """ ,len(np.unique(y)))

#  Now Selecting the best Parameters

def add_parameter(classifier1):
    param=dict()
    if classifier1=="knn":
        k=st.sidebar.slider("k value",1,15)
        # 1 is lowest k value and 15 is maximum k value
        param["k"]=k
    elif classifier1=="svm":
        # c is a regularisation parameter where strength of regularisation is iversely propotional to C
         c=st.sidebar.slider("C value",0.01,10.0)
         param["C"]=c
    else:
        # min_samples_leaf is minimum number of samples to be at leaf node
        # no of estimators-- number of trees in forest
         d=st.sidebar.slider("max_depth",2,10)
         n=st.sidebar.slider("No of Estimators",2,10)
         mi=st.sidebar.slider("min_samples_leaf",1,7)
         param["D"]=d
         param["N"]=n
         param["MI"]=mi
    return param
params=add_parameter(classifier)

def get_classifier(clf,params):
    if clf=="knn":
        model=KNeighborsClassifier(n_neighbors=params["k"])
       
    elif clf=="svm":

        model=SVC(C=params["C"]) # c is capital
    else:
       
        model=RandomForestClassifier(n_estimators=params["N"],max_depth=params["D"],min_samples_leaf=params["MI"],random_state=101)
    return model

model1=get_classifier(classifier,params)

# training starts now
st.write(params)
X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)
model1.fit(X_train,y_train)
y_pred=model1.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
st.write("accuracy with "+ classifier+" is ",accuracy)





