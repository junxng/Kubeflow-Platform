from kfp import dsl
from kfp import compiler
import kfp
from kfp.dsl import Input, Output, Dataset, Model, component
import os 

# Step 1: Load Dataset
@dsl.component(base_image="python:3.12")
def load_data(output_csv: Output[Dataset]):
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn"], check=True)

    from sklearn.datasets import load_iris
    import pandas as pd
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # Save the dataset to the output artifact path
    df.to_csv(output_csv.path, index=False)

# Step 2: Validate Dataset
@dsl.component(base_image="python:3.12")
def validate_dataset(
    input_csv: Input[Dataset],
    report: Output[Dataset]
):
    import pandas as pd
    
    df = pd.read_csv(input_csv.path)
    
    # Basic validation checks
    report_str = ""
    if df.isnull().values.any():
        report_str += "Dataset contains missing values.\\n"
    else:
        report_str += "Dataset has no missing values.\\n"
        
    if 'target' not in df.columns:
        report_str += "Dataset is missing the 'target' column.\\n"
    else:
        report_str += "Dataset has the 'target' column.\\n"
        
    with open(report.path, 'w') as f:
        f.write(report_str)

# Step 3: Preprocess Data
@dsl.component(base_image="python:3.12")
def preprocess_data(input_csv: Input[Dataset], output_train: Output[Dataset], output_test: Output[Dataset], 
                    output_ytrain: Output[Dataset], output_ytest: Output[Dataset]):
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn"], check=True)

    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Load dataset
    df = pd.read_csv(input_csv.path)

    # Handle missing values
    if df.isnull().values.any():
        df = df.dropna()  # Drop rows with any NaN values
    
    # Validate that there are no NaNs in the target column
    assert not df['target'].isnull().any(), "Target column contains NaN values after handling missing values."

    features = df.drop(columns=['target'])
    target = df['target']

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

    # Ensure no NaNs in the split data
    assert not y_train.isnull().any(), "y_train contains NaN values."
    assert not y_test.isnull().any(), "y_test contains NaN values."

    # Create DataFrames for train and test sets
    X_train_df = pd.DataFrame(X_train, columns=features.columns)
    y_train_df = pd.DataFrame(y_train) 
    X_test_df = pd.DataFrame(X_test, columns=features.columns)
    y_test_df = pd.DataFrame(y_test) 

    # Save processed train and test data
    X_train_df.to_csv(output_train.path, index=False)  
    X_test_df.to_csv(output_test.path, index=False)

    y_train_df.to_csv(output_ytrain.path, index=False)  
    y_test_df.to_csv(output_ytest.path, index=False) 

# Step 4: Hyperparameter Tuning
@dsl.component(
    base_image="python:3.12",
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
    
    model = LogisticRegression()
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    dump(grid_search.best_estimator_, best_model.path)

# Step 5: Train Model
@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "joblib",
        "boto3",
        "s3fs"
    ]
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
    import os
    from datetime import datetime
    import json
    import re

    # Validate required parameters
    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("AWS credentials are required. Use Kubernetes secrets or IAM roles.")
    if not s3_bucket or not s3_key:
        raise ValueError("S3 bucket and key are required parameters.")
    
    # Validate S3 bucket name format
    if not re.match(r'^[a-z0-9][a-z0-9.-]*[a-z0-9]$', s3_bucket):
        raise ValueError("Invalid S3 bucket name format")
    
    # Validate S3 key format
    if not re.match(r'^[a-zA-Z0-9/_-]+$', s3_key):
        raise ValueError("Invalid S3 key format")

    # Load the best model from hyperparameter tuning
    best_model = load(model_input.path)
    
    # Save the final model
    local_path = model_output.path
    dump(best_model, local_path)
    print(f"Model saved locally to: {local_path}")

    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        # Upload to S3
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_path = f"{s3_key}/model_{timestamp}.joblib"
        
        s3_client.upload_file(
            local_path,
            s3_bucket,
            s3_path
        )
        print(f"Model uploaded to s3://{s3_bucket}/{s3_path}")

        # Create outputs directory if it doesn't exist
        os.makedirs('/tmp/outputs', exist_ok=True)

        # Save S3 path to metadata
        metadata_path = '/tmp/outputs/output_metadata.json'
        model_uri = f"s3://{s3_bucket}/{s3_path}"
        with open(metadata_path, 'w') as f:
            json.dump({
                'model_s3_path': model_uri
            }, f)
        print(f"Metadata saved to: {metadata_path}")

        print(f"Returning model URI: {model_uri}")
        return model_uri  # Return the S3 URI as a string

    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        raise

# Step 6: Evaluate Model
@dsl.component(base_image="python:3.12")
def evaluate_model(test_data: Input[Dataset], ytest_data: Input[Dataset], model: Input[Model], metrics_output: Output[Dataset]) -> float:
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn", "matplotlib", "joblib"], check=True)

    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import matplotlib.pyplot as plt
    from joblib import load

    # Load test data
    X_test = pd.read_csv(test_data.path)

    y_test = pd.read_csv(ytest_data.path)  

    # Load model
    model = load(model.path)

    # Predict
    y_pred = model.predict(X_test)

    # Generate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred) 

    # Save metrics to a file
    metrics_path = metrics_output.path
    with open(metrics_path, 'w') as f:
        f.write(f"Accuracy: {accuracy}\\n")  # Add accuracy to the metrics file
        f.write(str(report))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(metrics_path.replace('.txt', '.png'))
    
    return accuracy

# Step 7: Deploy Model
@dsl.component(base_image="python:3.12")
def deploy_model(
    model_uri: str,
    deployment_name: str = "iris-classifier"
):
    # This is a placeholder for a real deployment process
    print(f"Deploying model from: {model_uri}")
    print(f"Deployment name: {deployment_name}")
    # In a real scenario, this would involve creating a prediction service
    # (e.g., a REST API with Flask/FastAPI) and deploying it to a serving
    # environment like Kubernetes or a cloud-based AI platform.
    pass

# Define the pipeline
@dsl.pipeline(name="ml-pipeline")
def ml_pipeline(
    aws_access_key_id: str = None,
    aws_secret_access_key: str = None,
    s3_bucket: str = None,
    s3_key: str = None
):
    # Step 1: Load Dataset
    load_op = load_data()

    # Step 2: Validate Dataset
    validate_op = validate_dataset(
        input_csv=load_op.outputs["output_csv"]
    )
    
    # Step 3: Preprocess Data
    preprocess_op = preprocess_data(input_csv=load_op.outputs["output_csv"])
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
    with dsl.Condition(evaluate_op.outputs['Output'] > 0.9): # Example threshold
        deploy_op = deploy_model(
            model_uri=train_op.outputs['Output']
        )
        deploy_op.after(evaluate_op)

# Compile the pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(pipeline_func=ml_pipeline, package_path="pipeline.yaml")