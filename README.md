# EHRSQL Introduction
This GitHub was created for CS-598 (Deep Learning for Healthcare) at UIUC for Spring 2023. EHRSQL aims to tackle question answering, and from those questions we generate SQL queries. The evaluations (found under the COMPLETE prefix folders) dictate how well it correctly recognizes as many possible answerable questions (while disregarding unanswerable queries) to gauge the overall model performance. The original GitHub that was used for this project can be found here: https://github.com/glee4810/EHRSQL.

**DISCLAIMER: Though it may look like the code files remain unchanged, I had to manually go through each file that is linked to one another and change its pathing directories to work for my situation. For whatever reason, the codes simply could not find their associated files, even after verifying the pathing directory to be correct. As a workaround, I changed the pathing directories for each file as necessary and placed all of the files in the same folder. It explains why the 'project' folder contains more files than organized folders.**

**In addition, the outputs folder does not exist for this GitHub because I am unable to download the massive 'checkpoint_best.pth.tar' without the Google Cloud machine terminating before completing the download. This file is necessary in order to generate the prediction_raw.json, which will be obtained after training has been completed. I have attempted to use gsutil to download objects from buckets, but I have insufficient permissions (most likely stemming from using solely a free trial) to perform this task. Thus, outputs folder has been omitted.**

# Acquiring Databases
First, have credentialed access to MIMIC-III and eiCU on PhysioNet. Once you have obtained access, download the MIMIC-III database from https://physionet.org/content/mimiciii/1.4/ and the eiCU database from https://physionet.org/content/eicu-crd/2.0/.

# Installation
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
Since this GitHub does not include the **MIMIC-III.db** or **eiCU.db** files, please download the aforementioned .db files and place them into the **project** folder. The database files will be used during evaluations as the final step. If you are using this GitHub, the data should already be preprocessed. If not, using the downloaded database files, perform the following commands:
```
cd preprocess
python3 preprocess_db.py --data_dir <path_to_mimic_iii_csv_files> --db_name mimic_iii --deid --timeshift --current_time "2105-12-31 23:59:00" --start_year 2100 --time_span 5 --cur_patient_ratio 0.1 &
python3 preprocess_db.py --data_dir <path_to_eicu_csv_files> --db_name eicu --deid --timeshift --current_time "2105-12-31 23:59:00" --start_year 2100 --time_span 5 --cur_patient_ratio 0.1
```

# Training and Generating Prediction Files
To note, **'t5_ehrsql_eicu_natural_lr0.001.yaml'** and **'t5_ehrsql_mimic3_natural_lr0.001.yaml'** are your TRAINING files. This will be used with the config.py file (modify the amount of steps, or other hyperparameters like optim and amount of workers for example) to perform training. 

Once 'x' amount of steps (x being whatever amount of steps that training should do before completion), we then use **'t5_ehrsql_eicu_natural_lr0.001_best__eicu_natural_valid.yaml'** and **'t5_ehrsql_mimic3_natural_lr0.001_best__mimic3_natural_valid.yaml'** for EVALUATION. For example:
```
***Total training time will vary (based on your current computational resources), but personally, 
with Google Cloud services, it took over a week to conduct the full 100000 steps for either eiCU or MIMIC-III***
nohup python3 main.py --config t5_ehrsql_mimic3_natural_lr0.001.yaml --CUDA_VISIBLE_DEVICES "" --device "cpu" &> generate_train_for_mimic3_no_schema.out &

***This will generate the prediction_raw.json, which will eventually be used to generate the prediction.json.***
nohup python3 main.py --config t5_ehrsql_mimic3_natural_lr0.001_best__mimic3_natural_valid.yaml --output_file prediction_raw.json --CUDA_VISIBLE_DEVICES "" --device "cpu" &> generate_pred_for_mimic3_no_schema.out &
```
The 'outputs' pathing should be automatically established from the given directory. The 'outputs' folder will store the **'eval_t5_ehrsql_eicu_natural_lr0.001_best__eicu_natural_valid', 'eval_t5_ehrsql_mimic3_natural_lr0.001_best__mimic3_natural_valid', 't5_ehrsql_eicu_natural_lr0.001', and 't5_ehrsql_mimic3_natural_lr0.001'** (what folders get generated ultimately depends on which database was used [if eiCU, then only eiCU related folders get created; otherwise, MIMIC-III related folders]).
 
As an additional note, nohup allows files to be ran in the background without the fear of the virtual machine terminating through timeout. If no .out filename is given after the &> characters: **&> generate_train_for_mimic3_no_schema.out &**. the default .out filename will simply be nohup.out. Here is how to view nohup files in real-time, and remove nohup files (referring to the examples above):
```
***View nohup.out files***
tail -f generate_pred_for_mimic3_no_schema.out

***generate_pred_for_mimic3_no_schema.out is an example. Simply replace this with your .out file***
rm -rf generate_pred_for_mimic3_no_schema.out
```
To clarify, the threshold values 0.14144589 and 0.22580192 were the default I used that stems from the authors' GitHub. The threshold value -1 is the actual default value that the code defaults to, and will generate probable evaluations, unlike -3. This default value is explicitly stated in **abstain_with_entropy.py**.

# Continuation and Evaluation
Once you have performed 'x' training steps, we can proceed to the next step. We can perform our SQL filtering at will by changing the threshold value. This filtering essentially performs the following task: if the question’s prediction confidence exceeds a given threshold, the resulting SQL query will not be generated and thus return a NULL value. This affects the final evaluation values.

```
***Using threshold values and producing prediction.json from prediction_raw.json***
nohup python3 abstain_with_entropy.py --infernece_result_path outputs/eval_t5_ehrsql_mimic3_natural_lr0.001_best__mimic3_natural_valid --input_file prediction_raw.json --output_file prediction.json --threshold 0.14144589 &> create_prediction_json_mimic3_no_schema.out &
nohup python3 abstain_with_entropy.py --infernece_result_path outputs/eval_t5_ehrsql_eicu_natural_lr0.001_best__eicu_natural_valid --input_file prediction_raw.json  --output_file prediction.json --threshold 0.22580192 &> create_prediction_json_eicu_no_schema.out &

```

We can perform evaluations either through giving --db_path a designated path, or both database files (eicu.db or mimic_iii.db) being found in the same folder.
```
***Performing evaluations with directory pathing to a database***
nohup python3 evaluate.py --db_path ./dataset/ehrsql/mimic_iii/mimic_iii.db --data_file dataset/ehrsql/mimic_iii/valid.json --pred_file ./outputs/eval_t5_ehrsql_mimic3_natural_lr0.001_best__mimic3_natural_valid/prediction.json &> eval_SQL.out &

***Performing evaluations with database files within a given folder***
nohup python3 evaluate.py --db_path eicu.db --data_file valid.json --pred_file ./outputs/eval_t5_ehrsql_eicu_natural_lr0.001_best__eicu_natural_valid/prediction.json &> eval_SQL_eicu_no_schema.out &
nohup python3 evaluate.py --db_path mimic_iii.db --data_file valid.json --pred_file ./outputs/eval_t5_ehrsql_mimic3_natural_lr0.001_best__mimic3_natural_valid/prediction.json &> eval_SQL_mimic3_no_schema.out &

```

<<<<<<< HEAD
To view these evaluations with varying threshold values, refer to the **'COMPLETE_eval_t5_ehrsql_eicu_natural_lr0.001_best__eicu_natural_valid'** and **'COMPLETE_eval_t5_ehrsql_mimic3_natural_lr0.001_best__mimic3_natural_valid'** folders for screenshots and their associated prediction_raw.json and prediction.json files.
=======
To view these evaluations with varying threshold values, refer to the **'COMPLETE_eval_t5_ehrsql_eicu_natural_lr0.001_best__eicu_natural_valid'** and **'COMPLETE_eval_t5_ehrsql_mimic3_natural_lr0.001_best__mimic3_natural_valid'** folders for screenshots and their associated prediction_raw.json and prediction.json files. You can experiment with other threshold values and see how the results vary.
>>>>>>> 07a40dd1c8f516f98b51337f3e653bd636cfdad9

# Credits

Authors for the EHRSQL paper: Gyubok Lee, Hyeonji Hwang, Seongsu Bae, Yeonsu Kwon, Woncheol Shin, Seongjun Yang, Minjoon Seo, Jong-Yeup Kim, Edward Choi

Link to EHRSQL paper: https://arxiv.org/abs/2301.07695
