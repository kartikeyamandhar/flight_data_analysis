
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('flight_delay_prediction').getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('ERROR')

import numpy as np 
import pandas as pd 

flights = spark.read.csv("s3://flight-0511/flights.csv",header=True)


flights.createOrReplaceTempView('flights')


flights = flights.withColumn('duration_hrs', flights.AIR_TIME/60)
flights = flights.withColumn("MONTH", flights.MONTH.cast("integer"))
flights = flights.withColumn("DAY_OF_WEEK", flights.DAY_OF_WEEK.cast("integer"))
flights = flights.withColumn("AIR_TIME", flights.AIR_TIME.cast("integer"))
flights = flights.withColumn("DISTANCE", flights.DISTANCE.cast("double"))
flights = flights.withColumn("ARRIVAL_DELAY", flights.ARRIVAL_DELAY.cast("integer"))

from pyspark.sql import functions as F
from pyspark.sql.functions import *


nof = flights.groupBy("TAIL_NUMBER").count()
nof.show()


nof1 = flights.groupBy("YEAR","ORIGIN_AIRPORT").count()


flights = flights.withColumn("DEPARTURE_DELAY",flights.DEPARTURE_DELAY.cast("integer"))
nof2 = flights.groupBy("MONTH","DESTINATION_AIRPORT")



airports = spark.read.csv("s3://flight-0511/airports.csv",header=True)
airlines = spark.read.csv("s3://flight-0511/airlines.csv",header=True)

airports = airports.withColumnRenamed("IATA_CODE", "DESTINATION_AIRPORT")

flights_with_airports = flights.join(airports , on = 'DESTINATION_AIRPORT', how = 'leftouter')

model_data = flights.select('MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'TAIL_NUMBER', 'DESTINATION_AIRPORT', 'AIR_TIME', 'DISTANCE', 'ARRIVAL_DELAY',)

model_data = model_data.filter("ARRIVAL_DELAY is not NULL and AIRLINE is not NULL and AIR_TIME is not NULL and TAIL_NUMBER is not NULL")



model_data = model_data.withColumn("is_late",model_data.ARRIVAL_DELAY > 0)
model_data = model_data.withColumn("is_late",model_data.is_late.cast("integer"))

model_data = model_data.withColumnRenamed("is_late", 'label')


# # LATE_LABEL DISTRIBUTION


model_data.groupBy("label").count().show()


# # ENCODING AND VECTORIZATION

from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

airline_indexer = StringIndexer(inputCol="AIRLINE", outputCol="airline_index")

# Create a OneHotEncoder
airline_encoder = OneHotEncoder(inputCol="airline_index", outputCol="airline_fact")


dest_indexer = StringIndexer(inputCol="DESTINATION_AIRPORT", outputCol="dest_index")

# Create a OneHotEncoder
dest_encoder = OneHotEncoder(inputCol="dest_index", outputCol="dest_fact")

tail_indexer = StringIndexer(inputCol="TAIL_NUMBER", outputCol="tail_index")

# Create a OneHotEncoder
tail_encoder = OneHotEncoder(inputCol="tail_index", outputCol="tail_fact")



from pyspark.ml.feature import VectorAssembler

# Make a VectorAssembler of 'MONTH', 'DAY_OF_WEEK', 'AIR_TIME', 'DISTANCE', 'ARRIVAL_DELAY','AIRLINE', 'TAIL_NUMBER', 'DESTINATION_AIRPORT'
vec_assembler = VectorAssembler(inputCols=["MONTH", "DAY_OF_WEEK", "AIR_TIME", "DISTANCE", "airline_fact", "dest_fact", "tail_fact"], outputCol="features")


from pyspark.ml import Pipeline

flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, airline_indexer, airline_encoder, tail_indexer, tail_encoder, vec_assembler])


piped_data = flights_pipe.fit(model_data).transform(model_data)

train_data, test_data = piped_data.randomSplit([.85, .15])

print('data points(rows) in train data :',  train_data.count())
print('data points(rows) in train data :',  test_data.count())


# # MODELLING


from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression()



import pyspark.ml.evaluation as evals

# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")


import pyspark.ml.tuning as tune

# Create the parameter grid
grid = tune.ParamGridBuilder()

# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])

# Build the grid
grid = grid.build()


# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator)

models = cv.fit(train_data)


best_lr = lr.fit(train_data)


# Use the model to predict the test set
test_results = best_lr.transform(test_data)

# Evaluate the predictions
print(evaluator.evaluate(test_results))

