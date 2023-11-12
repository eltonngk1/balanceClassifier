# balanceClassifier

#### Description:
Creates a stable or growth subset of users

#### Introduction:
In the banking industry, banks commonly use the funds deposited by customers to reap returns for profits, either by investing or loaning, keeping the remaining as reserves. Due to unpredictable financial conditions, it is difficult for banks to predict user transactional behaviour. As such, the motivation of this project lies in the goal of being able to segment customers based on their balance behaviour, identifying those who are growing (constantly depositing and increasing their total balance deposited) and those who are stable (relatively constant total balance). By doing so, more optimal financial decisions can be made on how to utilise the funds, and establish a distinct reserve rate. 

#### Directory:
```
balanceClassifier (project directory)
│   README.md
|   bestmodelselector.py
|   cfmodel.py
|   requirements.text
|   classification.ipynb
└───data
│   │   features.py
│   │   train_data.csv
│   │   test_data.csv
│   │   ...
│   
└───models
│   │   lgbm.py
│   │   xgb.py
│   │   logreg.py
│   │   mlp.py
│   │   stacking.py
│   │   voting.py
│   │   ...
│   
└───utils
│   │   calculations.py
│   │   model_helpers.py
│   │   ...
│   
```
#### Roles
- CfModel - base class for a classification model <br>
- BestModelSelector - supervisor class which instantiates the various classification models and handles plotting & comparison metrics

#### Configuration:
For further optimisation of model (only model parameters and user features): 
1) Go to the models folder and for each model.py file, edit the model parameters after running the gridsearch optimisation in the classification.ipynb file.
2) In the data folder, edit the features.py file for each of the models' feature list after running the gridsearch optimisation in the classification.ipynb file.

#### Steps to Replicate: 
1) Run this command: pip install -r requirements.txt
2) Run the subset_methodology.ipynb to get user labels for the 180 days data (i.e. user_subset_label.csv)
3) Run the feature_engineering.ipynb (read the user_subset_label.csv generated from step 2) to get the user features for the 90 days data for both stable (i.e. stable_feature.csv) and growth (i.e. growth_feature.csv)
4) Add your train_data.csv*, test_data.csv*, stable_feature.csv (generated from step 3), growth_feature.csv (generated from step 3) into data folder
5) Edit the main.py to change the file paths
6) Run this command: python main.py

*accessable from the google drive link in our final report 

#### Expected Outcome: 
Best Model for growth subset: model | growth rate | stability index <br>
[list of user_id] <br>

Best Model for stable subset:  model | growth rate | stability index <br> 
[list of user_id] <br>




