# Security Hardening Guide

## Overview
This guide provides secure configuration for the MLOps Kubeflow pipeline, addressing security concerns identified in the security audit.

## 🔒 Security Improvements

### 1. Removed Hardcoded Defaults
- **Before**: S3 bucket name and key had hardcoded defaults
- **After**: All sensitive parameters are now required with no defaults
- **File**: `secure-pipeline.py` replaces `pipeline.py`

### 2. Input Validation
- Added regex validation for S3 bucket names and keys
- Prevents injection attacks and invalid configurations
- Validates AWS credential presence

### 3. Secure Credential Management

#### Option A: Kubernetes Secrets (Recommended)
```bash
# Create secret from your credentials
kubectl create secret generic aws-credentials \
  --from-literal=aws-access-key-id=YOUR_ACCESS_KEY \
  --from-literal=aws-secret-access-key=YOUR_SECRET_KEY \
  --namespace=kubeflow

# Create ConfigMap for non-sensitive configuration
kubectl create configmap s3-config \
  --from-literal=s3-bucket=your-s3-bucket-name \
  --from-literal=s3-key=models/iris \
  --namespace=kubeflow
```

#### Option B: Environment Variables (Development)
```bash
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export S3_BUCKET=your-s3-bucket
export S3_KEY=models/iris
```

#### Option C: IAM Roles for Service Accounts (IRSA - Production)
```yaml
# Example IRSA configuration
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ml-pipeline-sa
  namespace: kubeflow
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT:role/ml-pipeline-role
```

## 🚀 Usage Instructions

### Secure Pipeline Execution

#### 1. Using Kubernetes Secrets
```bash
# Compile secure pipeline
python secure-pipeline.py

# Run pipeline with secrets
kfp pipeline create -p SecureIrisProject secure-pipeline.yaml

# When creating run, use secrets
kubectl get secret aws-credentials -n kubeflow -o jsonpath='{.data.aws-access-key-id}' | base64 -d
kubectl get secret aws-credentials -n kubeflow -o jsonpath='{.data.aws-secret-access-key}' | base64 -d
```

#### 2. Using Environment Variables
```bash
# Set required environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export S3_BUCKET="your-s3-bucket"
export S3_KEY="models/iris"

# Run secure pipeline
python secure-pipeline.py
```

#### 3. Using IRSA (Production)
```bash
# No credentials needed - uses IAM role
python secure-pipeline.py
```

## 🔐 Security Best Practices

### 1. AWS IAM Configuration
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-s3-bucket",
        "arn:aws:s3:::your-s3-bucket/*"
      ]
    }
  ]
}
```

### 2. Network Security
- Use VPC endpoints for S3 access
- Implement bucket policies with least privilege
- Enable S3 bucket encryption
- Use bucket versioning

### 3. Monitoring and Audit
- Enable CloudTrail for S3 access logging
- Set up CloudWatch alarms for unusual access patterns
- Implement AWS Config rules for compliance

## 📋 Security Checklist

- [ ] Remove all hardcoded credentials
- [ ] Use Kubernetes secrets for sensitive data
- [ ] Implement input validation
- [ ] Use IAM roles instead of access keys (production)
- [ ] Enable S3 bucket encryption
- [ ] Set up proper IAM policies
- [ ] Enable audit logging
- [ ] Use VPC endpoints for S3
- [ ] Implement network policies
- [ ] Regular security reviews

## 🛡️ Security Validation

### Validate S3 Parameters
```python
# Test bucket name validation
import re
bucket_pattern = r'^[a-z0-9][a-z0-9.-]*[a-z0-9]$'
test_buckets = [
    "valid-bucket-name",
    "invalid_bucket_name",
    "InvalidBucketName",
    "valid.bucket.name"
]

for bucket in test_buckets:
    is_valid = bool(re.match(bucket_pattern, bucket))
    print(f"{bucket}: {'✅ Valid' if is_valid else '❌ Invalid'}")
```

### Validate Credentials Setup
```bash
# Check if secrets are properly configured
kubectl get secret aws-credentials -n kubeflow -o yaml
kubectl get configmap s3-config -n kubeflow -o yaml
```

## 🔧 Troubleshooting

### Common Issues
1. **Invalid S3 bucket name**: Check regex pattern and AWS naming rules
2. **Access denied**: Verify IAM permissions and bucket policies
3. **Secrets not found**: Ensure secrets are in the correct namespace
4. **IRSA not working**: Check OIDC provider and IAM role trust relationships

### Debug Commands
```bash
# Check pod service account
kubectl describe pod <pod-name> -n kubeflow | grep "Service Account"

# Test S3 access from pod
kubectl exec -it <pod-name> -n kubeflow -- aws s3 ls

# Check IAM role (IRSA)
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/
```