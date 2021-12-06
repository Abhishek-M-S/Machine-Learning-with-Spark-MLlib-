
import sys
from sklearn.base import ClusterMixin
from sklearn.linear_model import Perceptron
import pickle
from pyspark.ml.feature import StandardScaler
import matplotlib.pyplot as plt
from pyspark.ml.feature import CountVectorizerModel
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from pyspark.mllib.linalg import Vector as MLLibVector, Vectors as MLLibVectors
from pyspark.sql.types import StringType,StructType,StructField
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.mllib.regression import LabeledPoint
from sklearn.preprocessing import MaxAbsScaler
from sklearn import preprocessing
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np
from sklearn.metrics import accuracy_score,precision_score
from pyspark.streaming import StreamingContext
from pyspark.sql import Row, SparkSession
from pyspark.ml.feature import HashingTF,IDF,Tokenizer,StringIndexer
from pyspark.ml.feature import RegexTokenizer,StopWordsRemover,CountVectorizer
from pyspark.ml.classification import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from pyspark.ml import Pipeline
from sklearn.metrics.pairwise import pairwise_distances_argmin
from pyspark.mllib.regression import StreamingLinearRegressionWithSGD
from sklearn import linear_model
from pyspark.ml.linalg import Vector
from pyspark.ml.functions import vector_to_array
from sklearn.cluster import MiniBatchKMeans
import json

def flatten(x):
    flattened_json_list=json.loads(x).values()
    for i in flattened_json_list:
        for key in i:
            i[key]=str(i[key])
    return(flattened_json_list)

sc = SparkContext("local[2]", "StreamingMachineLearning")
spark_context=SQLContext(sc)
ssc = StreamingContext(sc, 5)
lines=ssc.socketTextStream("localhost", 6100)
def vect():
    vec=HashingVectorizer(n_features=1000)
    return vec

def sca():
    scaler=MaxAbsScaler()
    return scaler

def passive():
    clif = linear_model.SGDClassifier()
    return clif
def process(time, rdd):
    try:
        if(rdd==[] or rdd is None or rdd==[[]]):
            return
        rdd=rdd.flatMap(lambda x:flatten(x))
        df=spark_context.createDataFrame(rdd,["subject","body","label"])

        X=df.select('body').collect()
        X=[row.body for row in X]
        vectorizer=vect()
        X=vectorizer.fit_transform(X)

        scaler=sca()
        X=scaler.fit_transform(X)

        y=df.select('label').collect()
        y=np.array([row[0] for row in np.array(y)])

        le = preprocessing.LabelEncoder()
        y=le.fit_transform(y)


        global count
        global ini
        global feature_extraction_pipeline
        
        clif=None

        if(ini):
            ini=False
            clif = passive()
            
        else:
            with open('./mod', 'rb') as p:
                clif = pickle.load(p)
            preds=clif.predict(X)
            score=accuracy_score(y,preds)
            print(count," ",score)
            count+=1
        clif.partial_fit(X,y,classes=np.unique([0,1]))

        pickle.dump(clif, open('./mod', 'wb'))
    except Exception as excep:
        print(str(excep))
ini=True
count=0

lines.foreachRDD(process)


ssc.start()
ssc.awaitTermination()