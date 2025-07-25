from kfp import dsl
from kfp import compiler
import kfp
from kfp.dsl import Input, Output, Dataset, Model, component
import os

# Step 1: Load Dataset
@dsl.component(base_image="python:3.12-slim", packages_to_install=["pandas", "scikit-learn"])
def load_data(output_csv: Output[Dataset]):
    from sklearn.datasets import load_iris
    import pandas as pd
    
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df.to_csv(output_csv.path, index=False)

# Step 2: Validate Dataset
@dsl.component(base_image="python:3.12-slim", packages_to_install=["pandas"])
def validate_dataset(input_csv: Input[Dataset], report: Output[Dataset]):
    import pandas as pd
    
    df = pd.read_csv(input_csv.path)
    
    report_lines = []
    if df.isnull().values.any():
        report_lines.append("Dataset contains missing values.")
    else:
        report_lines.append("Dataset has no missing values.")
        
    if 'target' not in df.columns:
        report_lines.append("Dataset is missing the 'target' column.")
    else:
        report_lines.append("Dataset has the 'target' column.")
        
    with open(report.path, 'w') as f:
        f.write('\n'.join(report_lines))

# Step 3: Preprocess Data
@dsl.component(base_image="python:3.12-slim", packages_to_install=["pandas", "scikit-learn"])
def preprocess_data(
    input_csv: Input[Dataset], 
    output_train: Output[Dataset], 
    output_test: Output[Dataset], 
    output_ytrain: Output[Dataset], 
    output_ytest: Output[Dataset]
):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_csv.path)
    
    if df.isnull().values.any():
        df = df.dropna()
    
    features = df.drop(columns=['target'])
    target = df['target']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, target, test_size=0.2, random_state=42
    )

    pd.DataFrame(X_train, columns=features.columns).to_csv(output_train.path, index=False)
    pd.DataFrame(X_test, columns=features.columns).to_csv(output_test.path, index=False)
    pd.DataFrame(y_train).to_csv(output_ytrain.path, index=False)
    pd.DataFrame(y_test).to_csv(output_ytest.path, index=False)

# Step 4: Hyperparameter Tuning
@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def hyperparameter_tuning(
    train_data: Input[Dataset],
    ytrain_data: Input[Dataset],
    best_model: Output[Model]
):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from joblib import dump

    X_train = pd.read_csv(train_data.path)
    y_train = pd.read_csv(ytrain_data.path).values.ravel()

    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'solver': ['liblinear', 'saga']
    }
    
    model = LogisticRegression(max_iter=1000)
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    dump(grid_search.best_estimator_, best_model.path)

# Step 5: Train Model
@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib", "boto3"]
)
def train_model(
    model_input: Input[Model],
    model_output: Output[Model],
    aws_access_key_id: str,
    aws_secret_access_key: str,
    s3_bucket: str,
    s3_key: str
) -> str:
    from joblib import load, dump
    import boto3
    import re
    from datetime import datetime
    import json

    # Input validation
    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("AWS credentials are required")
    if not s3_bucket or not s3_key:
        raise ValueError("S3 bucket and key are required")
    
    # Validate bucket name format
    if not re.match(r'^[a-z0-9][a-z0-9.-]*[a-z0-9]$', s3_bucket):
        raise ValueError("Invalid S3 bucket name format")
    
    # Validate key format
    if not re.match(r'^[a-zA-Z0-9/_-]+$', s3_key):
        raise ValueError("Invalid S3 key format")

    best_model = load(model_input.path)
    local_path = model_output.path
    dump(best_model, local_path)

    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_path = f"{s3_key}/model_{timestamp}.joblib"
        
        s3_client.upload_file(local_path, s3_bucket, s3_path)
        
        model_uri = f"s3://{s3_bucket}/{s3_path}"
        return model_uri

    except Exception as e:
        raise RuntimeError(f"Failed to upload to S3: {str(e)}")

# Step 6: Evaluate Model
@dsl.component(base_image="python:3.12-slim", packages_to_install=["pandas", "scikit-learn", "joblib"])
def evaluate_model(
    test_data: Input[Dataset], 
    ytest_data: Input[Dataset], 
    model: Input[Model], 
    metrics_output: Output[Dataset]
) -> float:
    import pandas as pd
    from sklearn.metrics import classification_report, accuracy_score
    from joblib import load

    X_test = pd.read_csv(test_data.path)
    y_test = pd.read_csv(ytest_data.path).values.ravel()
    model = load(model.path)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    with open(metrics_output.path, 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(str(report))
    
    return accuracy

# Step 7: Deploy Model
@dsl.component(base_image="python:3.12-slim")
def deploy_model(model_uri: str, deployment_name: str = "iris-classifier"):
    print(f"Deploying model from: {model_uri}")
    print(f"Deployment name: {deployment_name}")

# Define the pipeline
@dsl.pipeline(name="secure-ml-pipeline")
def secure_ml_pipeline(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    s3_bucket: str,
    s3_key: str
):
    # Validate required parameters
    if not all([aws_access_key_id, aws_secret_access_key, s3_bucket, s3_key]):
        raise ValueError("All parameters are required")

    # Step 1: Load Dataset
    load_op = load_data()

    # Step 2: Validate Dataset
    validate_op = validate_dataset(input_csv=load_op.outputs["output_csv"])
    
    # Step 3: Preprocess Data
    preprocess_op = preprocess_data(
        input_csv=load_op.outputs["output_csv"]
    )
    preprocess_op.after(validate_op)

    # Step 4: Hyperparameter Tuning
    hpt_op = hyperparameter_tuning(
        train_data=preprocess_op.outputs["output_train"],
        ytrain_data=preprocess_op.outputs["output_ytrain"]
    )

    # Step 5: Train Model
    train_op = train_model(
        model_input=hpt_op.outputs["best_model"],
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        s3_bucket=s3_bucket,
        s3_key=s3_key
    )

    # Step 6: Evaluate Model
    evaluate_op = evaluate_model(
        test_data=preprocess_op.outputs["output_test"],
        ytest_data=preprocess_op.outputs["output_ytest"],
        model=train_op.outputs["model_output"]
    )

    # Step 7: Deploy Model (conditionally)
    with dsl.Condition(evaluate_op.outputs['Output'] > 0.9):
        deploy_op = deploy_model(
            model_uri=train_op.outputs['Output']
        )
        deploy_op.after(evaluate_op)

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=secure_ml_pipeline, 
        package_path="secure-pipeline.yaml"
    )