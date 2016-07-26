#!/bin/bash
# Run on a Spark standalone cluster in cluster deploy mode with supervise

#environment variables
env=dev
log=/var/$env/pgm_log

#Spark variables
master_spark=spark://TYDEVANA1:7077
 # --deploy-mode cluster \
 # --supervise \
 # --executor-memory 20G \
 # --total-executor-cores 100 \
spark_submit=/var/$env/spark-1.6.2-bin-hadoop2.6/bin/spark-submit

export env
export log
export master_spark
export spark_submit