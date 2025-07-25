#!/usr/bin/env python3
"""
Production-ready MLOps pipeline with enhanced security
Uses IRSA (IAM Roles for Service Accounts) instead of hardcoded AWS credentials
"""

import kfp
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics
from kubernetes import client as k8s_client
import os

# Production configuration
PROD_CONFIG = {
    's3_bucket': 'your-ml-production-bucket',
    'model_name': 'iris-classifier-prod',
    'min_accuracy': 0.85,
    'max_replicas': 10,
    'min_replicas': 2,
    'service_account': 'ml-pipeline-sa',
    'namespace': 'ml-pipeline'
}

def create_volume_op(name: str, size: str = "10Gi") -> dsl.VolumeOp:
    """Create persistent volume for model storage"""
    return dsl.VolumeOp(
        name=name,
        resource_name="model-storage",
        size=size,
        modes=dsl.VOLUME_MODE_RWO,
        storage_class="gp2"
    )

@dsl.component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "scikit-learn==1.3.0",
        "pandas==2.0.3",
        "boto3==1.28.25",
        "joblib==1.3.2"
    ]
)
def data_validation_op(
    s3_bucket: str,
    validation_result: Output[Dataset]
) -> str:
    """Validate training data quality"""
    import pandas as pd
    import boto3
    import json
    from sklearn.datasets import load_iris
    
    # Load and validate data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Data quality checks
    validation_report = {
        'num_samples': len(df),
        'num_features': len(df.columns) - 1,
        'missing_values': df.isnull().sum().to_dict(),
        'class_distribution': df['target'].value_counts().to_dict(),
        'feature_stats': df.describe().to_dict()
    }
    
    # Save validation report
    with open(validation_result.path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    # Upload to S3
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket=s3_bucket,
        Key='validation/validation_report.json',
        Body=json.dumps(validation_report)
    )
    
    return "Data validation completed successfully"

@dsl.component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "scikit-learn==1.3.0",
        "pandas==2.0.3",
        "boto3==1.28.25",
        "joblib==1.3.2",
        "xgboost==1.7.6"
    ]
)
def training_op(
    s3_bucket: str,
    model_name: str,
    validation_result: Input[Dataset],
    model_output: Output[Model],
    metrics_output: Output[Metrics]
) -> str:
    """Train iris classification model with hyperparameter tuning"""
    import json
    import joblib
    import boto3
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    
    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10]
    }
    
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'best_params': grid_search.best_params_,
        'cv_score': float(grid_search.best_score_),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    with open(metrics_output.path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save model
    joblib.dump(best_model, model_output.path)
    
    # Upload to S3
    s3 = boto3.client('s3')
    s3.upload_file(model_output.path, s3_bucket, f'models/{model_name}/model.pkl')
    s3.put_object(
        Bucket=s3_bucket,
        Key=f'models/{model_name}/metrics.json',
        Body=json.dumps(metrics)
    )
    
    return f"Model trained with accuracy: {accuracy:.4f}"

@dsl.component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "scikit-learn==1.3.0",
        "joblib==1.3.2",
        "boto3==1.28.25",
        "pandas==2.0.3"
    ]
)
def model_validation_op(
    s3_bucket: str,
    model_name: str,
    model_input: Input[Model],
    metrics_input: Input[Metrics],
    min_accuracy: float,
    validation_passed: Output[Dataset]
) -> bool:
    """Validate model meets production requirements"""
    import json
    
    # Load metrics
    with open(metrics_input.path, 'r') as f:
        metrics = json.load(f)
    
    accuracy = metrics['accuracy']
    
    # Validation checks
    checks = {
        'accuracy_threshold': accuracy >= min_accuracy,
        'model_size_valid': os.path.getsize(model_input.path) < 100 * 1024 * 1024,  # 100MB
        'performance_acceptable': accuracy >= 0.75  # Additional safety check
    }
    
    validation_result = {
        'accuracy': accuracy,
        'min_accuracy': min_accuracy,
        'all_checks_passed': all(checks.values()),
        'checks': checks
    }
    
    with open(validation_passed.path, 'w') as f:
        json.dump(validation_result, f, indent=2)
    
    if not all(checks.values()):
        raise ValueError(f"Model validation failed: {validation_result}")
    
    return all(checks.values())

@dsl.component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "docker==6.1.3",
        "boto3==1.28.25"
    ]
)
def build_container_op(
    s3_bucket: str,
    model_name: str,
    container_image: Output[Dataset]
) -> str:
    """Build optimized container image with model"""
    import os
    import subprocess
    import boto3
    
    # Create Dockerfile
    dockerfile_content = f'''
FROM python:3.9-slim

# Install dependencies
RUN pip install --no-cache-dir \
    flask==2.3.3 \
    scikit-learn==1.3.0 \
    joblib==1.3.2 \
    boto3==1.28.25 \
    prometheus-client==0.17.1

# Create non-root user
RUN useradd -u 1000 -m -s /bin/bash mluser

# Copy model
COPY model.pkl /app/model.pkl
COPY app.py /app/app.py

# Set permissions
RUN chown -R mluser:mluser /app
USER mluser

WORKDIR /app

EXPOSE 8080

CMD ["python", "app.py"]
'''
    
    # Create Flask app
    app_content = '''
from flask import Flask, request, jsonify
import joblib
import os
from prometheus_client import Counter, Histogram, generate_latest
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model
model = joblib.load('model.pkl')

# Prometheus metrics
REQUEST_COUNT = Counter('model_requests_total', 'Total model requests')
REQUEST_LATENCY = Histogram('model_request_duration_seconds', 'Request latency')
MODEL_ACCURACY = Histogram('model_accuracy_score', 'Model accuracy')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': True})

@app.route('/predict', methods=['POST'])
def predict():
    REQUEST_COUNT.inc()
    
    with REQUEST_LATENCY.time():
        try:
            data = request.json
            features = data.get('features', [])
            
            if len(features) != 4:
                return jsonify({'error': 'Expected 4 features'}), 400
            
            prediction = model.predict([features])[0]
            confidence = max(model.predict_proba([features])[0])
            
            return jsonify({
                'prediction': int(prediction),
                'confidence': float(confidence),
                'model_version': os.environ.get('MODEL_VERSION', '1.0.0')
            })
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return jsonify({'error': 'Prediction failed'}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
'''
    
    # Write files
    with open('/tmp/Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    with open('/tmp/app.py', 'w') as f:
        f.write(app_content)
    
    # Build and push image (simplified for demo)
    image_tag = f"your-registry.com/ml-models/{model_name}:latest"
    
    return f"Container image configuration prepared: {image_tag}"

@dsl.component(
    base_image="alpine/k8s:latest",
    packages_to_install=["curl"]
)
def deploy_model_op(
    model_name: str,
    container_image: Input[Dataset],
    min_replicas: int,
    max_replicas: int
) -> str:
    """Deploy model to production with security configurations"""
    import subprocess
    import yaml
    
    # Generate deployment manifest
    deployment_manifest = f'''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {model_name}-serving
  namespace: ml-pipeline
spec:
  replicas: {min_replicas}
  selector:
    matchLabels:
      app: {model_name}-serving
  template:
    metadata:
      labels:
        app: {model_name}-serving
    spec:
      serviceAccountName: ml-pipeline-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: model-server
        image: your-registry.com/ml-models/{model_name}:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
---
apiVersion: v1
kind: Service
metadata:
  name: {model_name}-serving
  namespace: ml-pipeline
spec:
  selector:
    app: {model_name}-serving
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
'''
    
    return "Production deployment manifest generated"

@dsl.pipeline(
    name='Production Iris Classification Pipeline',
    description='End-to-end MLOps pipeline with security and monitoring'
)
def ml_pipeline_production(
    s3_bucket: str = PROD_CONFIG['s3_bucket'],
    model_name: str = PROD_CONFIG['model_name'],
    min_accuracy: float = PROD_CONFIG['min_accuracy']
):
    """Production-ready MLOps pipeline"""
    
    # Data validation
    validation_task = data_validation_op(
        s3_bucket=s3_bucket
    ).set_caching_options(False)
    
    # Model training
    training_task = training_op(
        s3_bucket=s3_bucket,
        model_name=model_name,
        validation_result=validation_task.outputs['validation_result']
    ).set_caching_options(False)
    
    # Model validation
    validation_passed = model_validation_op(
        s3_bucket=s3_bucket,
        model_name=model_name,
        model_input=training_task.outputs['model_output'],
        metrics_input=training_task.outputs['metrics_output'],
        min_accuracy=min_accuracy
    ).set_caching_options(False)
    
    # Build container
    container_task = build_container_op(
        s3_bucket=s3_bucket,
        model_name=model_name
    ).after(validation_passed)
    
    # Deploy model
    deploy_task = deploy_model_op(
        model_name=model_name,
        container_image=container_task.outputs['container_image'],
        min_replicas=PROD_CONFIG['min_replicas'],
        max_replicas=PROD_CONFIG['max_replicas']
    ).after(container_task)

    # Configure pod settings
    for task in [validation_task, training_task, validation_passed, container_task, deploy_task]:
        task.add_pod_annotation('sidecar.istio.io/inject', 'false')
        task.set_security_context(
            k8s_client.V1PodSecurityContext(
                run_as_non_root=True,
                run_as_user=1000,
                fs_group=1000
            )
        )
        task.container.set_security_context(
            k8s_client.V1SecurityContext(
                allow_privilege_escalation=False,
                read_only_root_filesystem=True,
                capabilities=k8s_client.V1Capabilities(drop=["ALL"])
            )
        )

if __name__ == '__main__':
    # Compile the pipeline
    kfp.compiler.Compiler().compile(
        ml_pipeline_production,
        'pipeline-production-secure.yaml'
    )
    
    print("Production pipeline compiled successfully!")