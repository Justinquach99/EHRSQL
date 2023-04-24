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
