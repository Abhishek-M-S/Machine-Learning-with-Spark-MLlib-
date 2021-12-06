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
from pyspark.ml import Pipeline
from sklearn.metrics.pairwise import pairwise_distances_argmin
from pyspark.mllib.regression import StreamingLinearRegressionWithSGD
from sklearn import linear_model
from pyspark.ml.linalg import Vector
from pyspark.ml.functions import vector_to_array
from sklearn.cluster import MiniBatchKMeans
import json



def vect():
    vec=HashingVectorizer(n_features=1000)
    return vec

def sca():
    scaler=MaxAbsScaler()
    return scaler

def flatten(x):
    json_list=json.loads(x).values()
    for i in json_list:
        for key in i:
            i[key]=str(i[key])
    return(json_list)

def max():
    clif = MiniBatchKMeans()
    return clif

def process(time, rd1):
    try:
        if(rd1==[] or rd1 is None or rd1==[[]]):
            return
        rd1=rd1.flatMap(lambda x:flatten(x))
        df=spark_context.createDataFrame(rd1,["subject","body","label"])

        X=df.select('body').collect()
        temp=[]
        for row in X:
            temp.append(row.body)
        X=temp
        vectorizer=vect()
        X_tr=vectorizer.fit_transform(X)
        scaler=sca()
        X_tr=scaler.fit_transform(X_tr)
        y_train=df.select('label').collect()
        y_train=np.array([row[0] for row in np.array(y_train)])

        le = preprocessing.LabelEncoder()
        y_train=le.fit_transform(y_train)

        global count
        global initial_run
        global feature_extraction_pipeline


        clif=None

        if(initial_run):
            initial_run=False
            clif = max()
        else:
            with open('./mod', 'rb') as p1:
                clif = pickle.load(p1)
            preds=clif.predict(X_tr)
            mbk_means = np.sort(clif.cluster_centers_, axis = 0)
            mbk_means_labels = pairwise_distances_argmin(X_tr, mbk_means)
            print(count," ",mbk_means_labels)
            count+=1
        clif.partial_fit(X_tr)
        pickle.dump(clif, open('./mod', 'wb'))
    except Exception as excep:
        print(str(excep))


sc = SparkContext("local[2]", "StreamingMachineLearning")
spark_context=SQLContext(sc)
ssc = StreamingContext(sc, 5)
lines=ssc.socketTextStream("localhost", 6100)

initial_run=True
count=0
model=LogisticRegression(featuresCol='features',labelCol='indexed_label',maxIter=10)
stage1=Tokenizer(inputCol="body",outputCol="words")
stage2=HashingTF(inputCol=stage1.getOutputCol(),outputCol="features1",numFeatures=1000)
stage3=StringIndexer(inputCol='label',outputCol='indexed_label')
scaler=StandardScaler()
scaler.setInputCol('features1')
scaler.setOutputCol('features')


pipeline=Pipeline(stages=[stage1,stage2,stage3,model])
feature_extraction_pipeline=Pipeline(stages=[stage1,stage2,scaler,stage3])

lines.foreachRDD(process)


ssc.start()
ssc.awaitTermination()