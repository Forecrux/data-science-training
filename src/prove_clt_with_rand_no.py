"""
Name: prove_clt_with_rand_no.py
Written by: Quinton Lai
Version: 1.0
Dependency: None
Description:
This program is to prove the central limit theorm with random number using SparkSQL, 
to approch the normal distribution, the program will repeat sampling with replacement
from the the original dataset.

Version history:

Programmer      Date        Version     Description
==========      =======     ========    ===========
Quinton         25Jul2016   1.0         Initial version

"""

from pyspark.sql.functions import rand

org_data = sql.Context.range(0,1000)
org_data.select("id", rand(seed=0).alias("y"))