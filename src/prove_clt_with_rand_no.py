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
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')   #Force the back-end without WindowX
import matplotlib.pyplot as plt

#Generate random sample space and its Q-Q plot
org_data = np.random.uniform(low = 0.0, high = 100.0, size = 250000)
org_data_qqplot = stats.probplot(org_data, dist = "norm", plot = plt)
plt.title("Normal Q-Q plot (Original dataset)")
plt.savefig("/var/dev/pgm_log/qqplot.png")