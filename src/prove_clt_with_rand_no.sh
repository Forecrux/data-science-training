env=dev
cd /var/$env/pgm

. ./ini_00.sh
pgm_root=/var/$env/pgm
pgm=prove_clt_with_rand_no
python3 $pgm_root/$pgm.py >$log/$pgm.log