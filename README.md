1. Project Overview
    - This is a Regression problem that has input features about rider, vehicle, weather conditions, traffic and pickup location and delivery location

2. Buisness use case


3. Impact of Solution


3. To evaluate a **Delivery Time Prediction** model effectively, **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)** can be used. These metrics help quantify the difference between predicted and actual delivery times, providing clear insight into model accuracy.


4. Project Setup
    - Create a project folder
    - git init
    - dvc init -f
    - Create venv
    - pip install -r requirements.txt
    - Create github repo and add remote: git remote add origin <.git url>
    - Run template.py to create folder and files
    - Create setup.py and pip install -e .


    - Setup s3 buket & aws ecr for docker app
        - create iam user
        - attach policies
        - setup aws: pip install awscli && aws configure (access key and security key)


    - dvc remote add -d myremote s3://my-mlops-project-demo/house_price_prediction
    - dvc config core.autostage true ---> auto pull everytime
    - Setup dagshub and host mlflow 


5. Data Gathering

6. Data Assesment

7. Data Cleaning

5. EDA

6. Data Transformation & Feature Engineering

7. Model Training
    1. Baseline model
    2. Best Algorithm
    3. Hyper-parameter Tunining


8. Convert best overall into dvc pipeline: create components like data_ingestion, data_transformation, model_trainer, model_evalutaion, logger etc

9. Upload best model on dagshub hosted mlflow model_registry in staging 

10. Perform model test & if success then push model to production

11. Create fastapi (fetch latest model in production from mlflow model_registry and make predictions)

12. Create CICD Pipeline