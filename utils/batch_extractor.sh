#/bin/bash
echo 'hello'

dirname="../runs/curve"

mkdir -p $dirname

#python tensorboard_extract.py --in-path ../runs/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed888/logger --ex-path $dirname/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed888.csv
python tensorboard_extract.py --in-path ../runs/bottle_cap_cnn/seed111/logger --ex-path $dirname/bottle_cap_cnn/seed111.csv
python tensorboard_extract.py --in-path ../runs/bottle_cap_cnn/seed222/logger --ex-path $dirname/bottle_cap_cnn/seed222.csv
python tensorboard_extract.py --in-path ../runs/bottle_cap_cnn/seed333/logger --ex-path $dirname/bottle_cap_cnn/seed333.csv
python tensorboard_extract.py --in-path ../runs/bottle_cap_cnn/seed444/logger --ex-path $dirname/bottle_cap_cnn/seed444.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed555/logger --ex-path $dirname/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed555.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed666/logger --ex-path $dirname/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed666.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed777/logger --ex-path $dirname/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed777.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_vt20t-retac-tmr05-bin-ft+dataset-BottleCap/seed888/logger --ex-path $dirname/bottle_cap_vt20t-retac-tmr05-bin-ft+dataset-BottleCap/seed888.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_v-repic-t20-retac-ft-mr075-fr-mse-tmr05-bin-ft+BottleCap/seed888/logger --ex-path $dirname/bottle_cap_v-repic-t20-retac-ft-mr075-fr-mse-tmr05-bin-ft+BottleCap/seed888.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_v-repic-bin-ft-mr075-fr-mse+BottleCap/seed777/logger --ex-path $dirname/bottle_cap_v-repic-bin-ft-mr075-fr-mse+BottleCap/seed777.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_v-repic-bin-ft-mr075-fr-mse+BottleCap/seed888/logger --ex-path $dirname/bottle_cap_v-repic-bin-ft-mr075-fr-mse+BottleCap/seed888.csv



#python tensorboard_extract.py --in-path ../runs/0912/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed231/logger --ex-path $dirname/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap1/seed231.csv
#python tensorboard_extract.py --in-path ../runs/0912/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed333/logger --ex-path $dirname/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap1/seed333.csv
#python tensorboard_extract.py --in-path ../runs/0912/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed42/logger --ex-path $dirname/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap1/seed42.csv
#python tensorboard_extract.py --in-path ../runs/0912/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed444/logger --ex-path $dirname/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap1/seed444.csv
#python tensorboard_extract.py --in-path ../runs/0912/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed555/logger --ex-path $dirname/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap1/seed555.csv
#python tensorboard_extract.py --in-path ../runs/0912/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed666/logger --ex-path $dirname/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap1/seed666.csv
#python tensorboard_extract.py --in-path ../runs/0912/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed777/logger --ex-path $dirname/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap1/seed777.csv
#python tensorboard_extract.py --in-path ../runs/0912/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed888/logger --ex-path $dirname/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap1/seed888.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_v-repic-bin-ft-mr075-fr-mse+BottleCap/seed123/logger --ex-path $dirname/bottle_cap_v-repic-bin-ft-mr075-fr-mse+BottleCap/seed123.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_vmvp/seed123/logger --ex-path $dirname/bottle_cap_vmvp/seed123.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_vt20t-reall-bin-ft+dataset-BottleCap/seed123/logger --ex-path $dirname/bottle_cap_vt20t-reall-bin-ft+dataset-BottleCap/seed123.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed123/logger --ex-path $dirname/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap/seed123.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_vt20t-repic-bin-ft+dataset-BottleCap/seed123/logger --ex-path $dirname/bottle_cap_vt20t-repic-bin-ft+dataset-BottleCap/seed123.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_vt20t-repic-tmr05-bin-ft+dataset-BottleCap/seed123/logger --ex-path $dirname/bottle_cap_vt20t-repic-tmr05-bin-ft+dataset-BottleCap/seed123.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_vt20t-retac-bin-ft+dataset-BottleCap/seed123/logger --ex-path $dirname/bottle_cap_vt20t-retac-bin-ft+dataset-BottleCap/seed123.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_vt20t-retac-tmr025-bin-ft+dataset-BottleCap/seed123/logger --ex-path $dirname/bottle_cap_vt20t-retac-tmr025-bin-ft+dataset-BottleCap/seed123.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_vt20t-retac-tmr05-bin-ft+dataset-BottleCap/seed123/logger --ex-path $dirname/bottle_cap_vt20t-retac-tmr05-bin-ft+dataset-BottleCap/seed123.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_vt5t-reall-bin-ft+dataset-BottleCap/seed123/logger --ex-path $dirname/bottle_cap_vt5t-reall-bin-ft+dataset-BottleCap/seed123.csv
#python tensorboard_extract.py --in-path ../runs/bottle_cap_vt5t-repic-bin-ft+dataset-BottleCap/seed123/logger --ex-path $dirname/bottle_cap_vt5t-repic-bin-ft+dataset-BottleCap/seed123.csv