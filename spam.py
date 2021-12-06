from pyspark import SparkContext
from sklearn.feature_extraction.text import HashingVectorizer
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType
from pyspark.streaming import StreamingContext
import json
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression,NaiveBayes
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
import numpy as np
from sklearn.naive_bayes import MultinomialNB

def rdd1():
    spark=SparkSession.builder\
    .appName("Example")\
    .master("local")\
    .getOrCreate()
    
    
    listrdd = rdd1.collect()
    if(listrdd == [] or listrdd is None or listrdd == [[]]):
        return
    rd1=[]
    temp=[]
    for i in listrdd:
        rd1.append(json.loads(i).values())
    for i in rd1:
        for x in i:
            temp.append(x.values())
        rd1.append(temp)
    count=0
    lis=[]
    for i in rd1:
        if count<3:
            lis.append(i[count])
            count+=1
        else:
            break
    lis=np.array(lis)
    subject = lis[:,0]
    
    
    X_msg = lis[:,1]
    Y = lis[:,2]
    schema= StructType([
        StructField('Message',StringType(),True),
        StructField('Spam/Ham',StringType(),True)
    ])
    data=np.concatenate(X_msg,Y)
    rdd=spark.sparkContext.parallelize(data)
    df=spark.createDataFrame(rdd,schema)
    X,y=preproce(df)
    
    naivie_bays(X,y)
    Berno(X,y)
    linear(X,y)


def preproce(df):
    X=df.select('Message').collect()
    x=[i['Message'] for i in X]
    vectorizer=HashingVectorizer(n_features=100)
    X=vectorizer.fit_transform(X)
    y=df.select('Spam/Ham').collect()
    y=np.array([i[0] for i in np.array(y)])
    return(X,y)

def naivie_bays(X,y):
    clif=MultinomialNB()
    clif.fit(X,y)
    print(clif.predict(X[2]))

def Berno(X,y):
    clif=BernoulliNB()
    clif.fit(X,y)
    print(clif.predict(X[2]))

def linear(X,y):
    clf = linear_model.SGDRegressor()
    clf.fit(X, y)
    print(clf.predict(X[2]))


sc = SparkContext(master="local[100]", appName="streamtest")
ssc = StreamingContext(sc,10)
lines = ssc.socketTextStream("localhost", 6100)

lines.foreachRDD(rdd1)




ssc.start()
ssc.awaitTermination()

