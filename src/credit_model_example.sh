env=dev
cd /var/$env/pgm

. ./ini_00.sh
pgm_root=/var/$env/pgm
pgm=credit_model_example
$spark_submit --packages com.databricks:spark-csv_2.11:1.4.0 $pgm_root/$pgm.py >$log/$pgm.log