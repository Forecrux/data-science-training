"""
Name: credit_model_example.py
Version: 0.1

Description:
The aim of this model is to identify the bad customers
Datamap is on the page 3 of dmahmeq_datamap.pdf

Analytic approach:
    Data processing:
        All data is cleaned before being employed
        Bootstrap (Not performed yet)
        Vectorized data for machine learning
            Multilayer Percetron Classifier (Done)
            Decision Tree (Not yet)
            Logistic Regression (Not yet)
    Preliminary analysis:
        Chi-square analysis among predictors (Not yet)
    Modeling:
        Multilayer Percetron Classifier
        Decision Tree (In progress)
        Logistic Regression (Not yet)
    Model evaluation:
        Confusion matrix (Not yet)
        ROC analysis (Not yet)
        Lift value analysis (Not yet)
"""

from __future__ import print_function
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer
import sys

if __name__ == "__main__":

    sc = SparkContext(appName="Home-equity loan credit risk model from SAS")
    sqlContext = SQLContext(sc)
    
    #Output setting
    saveout = sys.stdout
    outfile = open('/var/dev/pgm_log/credit_model_example.out', 'w')
    sys.stdout = outfile

    #################################################################
    # Data processing
    #################################################################

    # Define schema
    customSchema = StructType([ \
    StructField("bad", DoubleType(), True), \
    StructField("reason", StringType(), True), \
    StructField("job", StringType(), True), \
    StructField("loan", DoubleType(), True), \
    StructField("mortdue", DoubleType(), True), \
    StructField("value", DoubleType(), True), \
    StructField("yoj", FloatType(), True), \
    StructField("derog", DoubleType(), True), \
    StructField("delinq", DoubleType(), True), \
    StructField("clage", FloatType(), True), \
    StructField("ninq", DoubleType(), True), \
    StructField("clno", DoubleType(), True), \
    StructField("debtinc", FloatType(), True)])

    # Load csv data
    riskdata = sqlContext.read \
    .format('com.databricks.spark.csv') \
    .options(header='true') \
    .load('file:///var/dev/raw_data/dmahmeq_mod.csv', schema = customSchema)

    #Drop string variables
    string_list = ['reason', 'job']
    clean_riskdata = riskdata.select([col for col in riskdata.columns if col not in string_list])
    
    #Handle missing value
    """
    clean_riskdata = nostring_riskdata.na.fill({"loan":avg("loan") ,
                                                "mortdue":avg("mortdue"), 
                                                "value":avg("value"),
                                                "derog":avg("derog"),
                                                "delinq":0,
                                                "clage":avg("clage"),
                                                "ninq":avg("ninq"),
                                                "clno":avg("clno"),
                                                "debtinc":avg("debtinc")
                                                })
    """
    #Define Input-output columns, i.e. transform to MLP features vector
    ignore=['bad']
    assembler = VectorAssembler(
    inputCols=[k for k in clean_riskdata.columns if k not in ignore],
    outputCol="predictors")
    Triskdata = assembler.transform(clean_riskdata)
    # Split the data into train and test
    splits = Triskdata.randomSplit([0.4, 0.6], 1234)
    train = splits[0]
    test = splits[1]

    #################################################################
    # Multilayer Perceptron Classifier
    #################################################################

    # specify layers for the neural network:
    # input layer of size 10 (features), two intermediate of size 3 and 2
    # and output of size 2 (classes)
    layers = [10, 3, 2, 2]
    # create the trainer and set its parameters
    MLPtrainer = MultilayerPerceptronClassifier(maxIter = 100, layers = layers,
                                             labelCol = "bad", featuresCol = "predictors",
                                             predictionCol = "prediction", 
                                             blockSize = 1000, seed = 1234)
    # train the model
    MLP_model = MLPtrainer.fit(train)
    
    # compute precision on the test set
    MLP_result = MLP_model.transform(test)
    MLP_predictionAndLabels = MLP_result.select("prediction", "bad")
    MLP_evaluator = MulticlassClassificationEvaluator(metricName="precision")
    print(MLP_model)
    print(str(MLP_result.show())) # Print first 20 rows result to output file (plain text)


    #################################################################
    # Decision Tree Classification
    #################################################################
    # Train a DecisionTree model.
    dt_model_spec = DecisionTreeClassifier(labelCol="bad", featuresCol="predictors")

    # Chain indexers and tree in a Pipeline
    dt_pipeline = Pipeline(stages=[bad, predictors, dt_model_spec])

    # Train model.  This also runs the indexers.
    dt_model = dt_pipeline.fit(train)

    # Make predictions.
    dt_predictions = model.transform(test)

    # Select example rows to display.
    dt_predictions.select("prediction", "bad", "predictors").show(5)

    # Select (prediction, true label) and compute test error
    dt_evaluator = MulticlassClassificationEvaluator(
        labelCol="bad", predictionCol="prediction", metricName="precision")
    dt_accuracy = evaluator.evaluate(dt_predictions)
    print("Test Error = %g " % (1.0 - accuracy))

    treeModel = dt_model.stages[2]
    # summary only
    print(treeModel)

    #Output setting
    sys.stdout = saveout
    outfile.close()
        
    sc.stop()
