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

#### Configuration:
For further optimisation of model (only model parameters and user features): 
1) Go to the models folder and for each model.py file, edit the model parameters after running the gridsearch optimisation on the notebook file. 
2) In the data folder, edit the features.py file for each of the models' feature list after running the gridsearch optimisation on the notebook file.

#### Steps to Replicate: 
1) Run this command: pip install -r requirements.txt
2) Add your train_data.csv and test_data.csv into data folder
3) Edit the main.py to change the file path of the train_data.csv and test_data.csv
4) Run this command: python main.py

#### Expected Outcome: 
Best Model for growth subset: model | growth rate | stability index
Best Model for stable subset:  model | growth rate | stability index



