# EHRSQL
This GitHub was created for CS-598 (Deep Learning for Healthcare) at UIUC for Spring 2023. EHRSQL aims to tackle question answering, and from those questions we generate SQL queries. The evaluations (found under the COMPLETE prefix folders) dictate how well it correctly recognizes as many possible answerable questions (while disregarding unanswerable queries) to gauge the overall model performance. The original GitHub that was used for this project can be found here: https://github.com/glee4810/EHRSQL.

# Acquiring Databases
First, have credentialed access to MIMIC-III and eiCU on PhysioNet. Once you have obtained access, download the MIMIC-III database from https://physionet.org/content/mimiciii/1.4/ and the eiCU database from https://physionet.org/content/eicu-crd/2.0/.

#Installation
Ensure pip is installed for your machine. These tasks should be completed before running codes:
```
git clone https://github.com/glee4810/EHRSQL.git
cd EHRSQL
conda create -n ehrsql python=3.7
conda activate ehrsql
pip install pandas
pip install dask
pip install wandb # if needed
pip install nltk
pip install scikit-learn
pip install func-timeout
```
These are other libraries that are needed but were not listed on the original GitHub:
```
sudo apt-get update
sudo apt-get install python3-pip

pip install sentencepiece
pip install transformers
pip install pyyaml
pip install torch
pip install tqdm
pip install numpy
```

# Preprocessing
If you are using this GitHub, the data should already be preprocessed. If not, using the downloaded database files, perform the following commands:
```
cd preprocess
python3 preprocess_db.py --data_dir <path_to_mimic_iii_csv_files> --db_name mimic_iii --deid --timeshift --current_time "2105-12-31 23:59:00" --start_year 2100 --time_span 5 --cur_patient_ratio 0.1 &
python3 preprocess_db.py --data_dir <path_to_eicu_csv_files> --db_name eicu --deid --timeshift --current_time "2105-12-31 23:59:00" --start_year 2100 --time_span 5 --cur_patient_ratio 0.1
```

# Training
To note, 't5_ehrsql_eicu_natural_lr0.001.yaml' and 't5_ehrsql_mimic3_natural_lr0.001.yaml' are your TRAINING files. This will be used with the config.py file (modify the amount of steps, or other hyperparameters like optim and amount of workers for example) to perform training. 

Once 'x' amount of steps (x being whatever amount of steps that training should do before completion), we then use 't5_ehrsql_eicu_natural_lr0.001_best__eicu_natural_valid.yaml' and 't5_ehrsql_mimic3_natural_lr0.001_best__mimic3_natural_valid.yaml' for EVALUATION. For example:
```
***Total training time will vary (based on your current computational resources), but personally, 
with Google Cloud services, it took over a week to conduct the full 100000 steps for either eiCU or MIMIC-III***
nohup python3 main.py --config t5_ehrsql_mimic3_natural_lr0.001.yaml --CUDA_VISIBLE_DEVICES "" --device "cpu" &> generate_train_for_mimic3_no_schema.out &

***This will generate the prediction_raw.json, which will eventually be used to generate the prediction.json.***
nohup python3 main.py --config t5_ehrsql_mimic3_natural_lr0.001_best__mimic3_natural_valid.yaml --output_file prediction_raw.json --CUDA_VISIBLE_DEVICES "" --device "cpu" &> generate_pred_for_mimic3_no_schema.out &
```
 The 'outputs' pathing should be automatically established from the given directory. The 'outputs' folder will store the 'eval_t5_ehrsql_eicu_natural_lr0.001_best__eicu_natural_valid', 'eval_t5_ehrsql_mimic3_natural_lr0.001_best__mimic3_natural_valid', 't5_ehrsql_eicu_natural_lr0.001', and 't5_ehrsql_mimic3_natural_lr0.001' (what folders get generated ultimately depends on which database was used [if eiCU, then only eiCU related folders get created; otherwise, MIMIC-III related folders]).
