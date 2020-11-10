import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg
from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression



import pandas as pd


spark = SparkSession.builder.master("local[4]")\
       .appName("test").getOrCreate()
df=spark.read.csv('train.csv',header=True,sep= ",",inferSchema=True)

df = df.na.drop()
df = df.withColumn("gender",when(df["gender"]=='M',0).otherwise(1))
df.groupBy('churnIn3Month').count().show()
df.select("phoneBalance","churnIn3Month").\
          groupBy("ChurnIn3Month").agg(avg("phoneBalance"))

from pyspark.ml.stat import Correlation
x=df.columns[2:11]
corr_plot = pd.DataFrame()
for i in x:
    corr=[]
    for j in x:
        corr.append(round(df.stat.corr(i,j),2))
    corr_plot = pd.concat([corr_plot,pd.Series(corr)],axis=1)
corr_plot.columns=x
corr_plot.insert(0,'',x)
corr_plot.set_index('')

ignore=['churnIn3Month', 'ID','_c0']
vectorAssembler = VectorAssembler(inputCols=[x for x in df.columns
                  if x not in ignore], outputCol = 'features')
new_df = vectorAssembler.transform(df)
new_df = new_df.select(['features', 'churnIn3Month'])

## train test split
train, test = new_df.randomSplit([0.75, 0.25], seed = 12345)

## logistic regression
lr = LogisticRegression(featuresCol = 'features',
                         labelCol='churnIn3Month')
lr_model = lr.fit(train)


lr_model.summary.areaUnderROC


## random forest classfier

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol =
                            'churnIn3Month')
rf_model = rf.fit(train)


from pyspark.ml.evaluation import BinaryClassificationEvaluator
predictions = rf_model.transform(test)
auc = BinaryClassificationEvaluator().setLabelCol('churnIn3Month')
print('AUC of the model:' + str(auc.evaluate(predictions)))


