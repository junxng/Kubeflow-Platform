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

# Step 2: Preprocess Data
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

# Step 3: Train Model
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
    train_data: Input[Dataset], 
    ytrain_data: Input[Dataset], 
    model_output: Output[Model],
    aws_access_key_id: str,
    aws_secret_access_key: str,
    s3_bucket: str,
    s3_key: str
) -> str:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from joblib import dump
    import boto3
    import os
    from datetime import datetime
    import json

    # Load training data
    train_df = pd.read_csv(train_data.path)
    X_train = train_df 

    y_train = pd.read_csv(ytrain_data.path)
    y_train = y_train.values.ravel()  # Fix the column-vector warning 

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # First save model locally
    local_path = model_output.path
    dump(model, local_path)
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

# Step 4: Evaluate Model
@dsl.component(base_image="python:3.12")
def evaluate_model(test_data: Input[Dataset], ytest_data: Input[Dataset], model: Input[Model], metrics_output: Output[Dataset]):
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
        f.write(f"Accuracy: {accuracy}\n")  # Add accuracy to the metrics file
        f.write(str(report))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(metrics_path.replace('.txt', '.png'))


# Define the pipeline
@dsl.pipeline(name="ml-pipeline")
def ml_pipeline(
    aws_access_key_id: str = os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key: str = os.getenv('AWS_SECRET_ACCESS_KEY'),
    s3_bucket: str = "kubeflow-bucket-dungnq49",
    s3_key: str = "models/iris"
):
    # Step 1: Load Dataset
    load_op = load_data()

    # Step 2: Preprocess Data
    preprocess_op = preprocess_data(input_csv=load_op.outputs["output_csv"])

    # Step 3: Train Model
    train_op = train_model(
        train_data=preprocess_op.outputs["output_train"],
        ytrain_data=preprocess_op.outputs["output_ytrain"],
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        s3_bucket=s3_bucket,
        s3_key=s3_key
    )

    # Step 4: Evaluate Model
    evaluate_op = evaluate_model(
        test_data=preprocess_op.outputs["output_test"],
        ytest_data=preprocess_op.outputs["output_ytest"],
        model=train_op.outputs["model_output"]
    )

# Compile the pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(pipeline_func=ml_pipeline, package_path="pipeline.yaml")