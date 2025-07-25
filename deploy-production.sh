#!/bin/bash
# Production deployment script for secure MLOps pipeline

set -e

echo "🚀 Starting production deployment..."

# Configuration
NAMESPACE="ml-pipeline"
CLUSTER_NAME="your-eks-cluster"
REGION="us-west-2"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
S3_BUCKET="your-ml-production-bucket"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

echo_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

echo_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo_error "AWS CLI is not installed"
    exit 1
fi

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo_error "kubectl is not installed"
    exit 1
fi

# Check eksctl
if ! command -v eksctl &> /dev/null; then
    echo_error "eksctl is not installed"
    exit 1
fi

echo_success "Prerequisites check passed"

# Create S3 bucket if it doesn't exist
echo "📦 Setting up S3 bucket..."
if ! aws s3 ls "s3://$S3_BUCKET" 2>/dev/null; then
    aws s3 mb "s3://$S3_BUCKET" --region $REGION
    aws s3api put-bucket-versioning --bucket $S3_BUCKET --versioning-configuration Status=Enabled
    aws s3api put-bucket-encryption --bucket $S3_BUCKET --server-side-encryption-configuration '{
        "Rules": [
            {
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "AES256"
                }
            }
        ]
    }'
    echo_success "S3 bucket created: $S3_BUCKET"
else
    echo_success "S3 bucket already exists: $S3_BUCKET"
fi

# Create IRSA role
echo "🔐 Setting up IRSA (IAM Roles for Service Accounts)..."

# Create IAM OIDC provider
eksctl utils associate-iam-oidc-provider --cluster $CLUSTER_NAME --region $REGION --approve

# Create IAM policy for S3 access
cat > ml-pipeline-s3-policy.json <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::$S3_BUCKET/*",
                "arn:aws:s3:::$S3_BUCKET"
            ]
        }
    ]
}
EOF

aws iam create-policy \
    --policy-name ml-pipeline-s3-policy \
    --policy-document file://ml-pipeline-s3-policy.json || echo_warning "Policy already exists"

# Create service account with IRSA
echo "🔗 Creating service account with IRSA..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

eksctl create iamserviceaccount \
    --cluster $CLUSTER_NAME \
    --region $REGION \
    --name ml-pipeline-sa \
    --namespace $NAMESPACE \
    --attach-policy-arn arn:aws:iam::$ACCOUNT_ID:policy/ml-pipeline-s3-policy \
    --approve \
    --override-existing-serviceaccounts

echo_success "IRSA setup completed"

# Apply security configurations
echo "🛡️ Applying security configurations..."
kubectl apply -f k8s/rbac-network-policies.yaml
kubectl apply -f k8s/resource-limits.yaml
echo_success "Security configurations applied"

# Deploy monitoring stack
echo "📊 Deploying monitoring stack..."
kubectl apply -f k8s/monitoring-stack.yaml
echo_success "Monitoring stack deployed"

# Deploy production resources
echo "🏗️ Deploying production resources..."
kubectl apply -f k8s/production-deployment.yaml
echo_success "Production resources deployed"

# Compile and deploy pipeline
echo "🔨 Compiling production pipeline..."
python pipeline-production-secure.py

echo "📋 Uploading pipeline to Kubeflow..."
# Upload pipeline (requires kfp CLI)
if command -v kfp &> /dev/null; then
    kfp pipeline upload -p "Iris Production Pipeline" pipeline-production-secure.yaml
    echo_success "Pipeline uploaded to Kubeflow"
else
    echo_warning "kfp CLI not found. Pipeline saved as pipeline-production-secure.yaml"
fi

# Verify deployment
echo "🔍 Verifying deployment..."
kubectl wait --for=condition=available --timeout=300s deployment/ml-metrics-exporter -n $NAMESPACE || true
kubectl wait --for=condition=available --timeout=300s deployment/iris-model-serving -n $NAMESPACE || true

echo ""
echo_success "🎉 Production deployment completed successfully!"
echo ""
echo "📊 Access points:"
echo "  - Model API: http://iris-model-serving.$NAMESPACE.svc.cluster.local"
echo "  - Metrics: http://ml-metrics-exporter.$NAMESPACE.svc.cluster.local:8080/metrics"
echo ""
echo "🔐 Security features enabled:"
echo "  - IRSA for AWS credential management"
echo "  - RBAC with least privilege"
echo "  - Network policies"
echo "  - Pod security contexts"
echo "  - Resource quotas and limits"
echo "  - Monitoring and alerting"
echo ""
echo "📚 Next steps:"
echo "  1. Update S3_BUCKET variable in deploy-production.sh"
echo "  2. Configure your domain for ingress"
echo "  3. Set up cert-manager for TLS"
echo "  4. Configure Grafana dashboards"
echo "  5. Set up alerting rules"
echo ""
echo "🚀 Ready to run: kfp run submit -e default -r iris-production-run"

# Cleanup temporary files
rm -f ml-pipeline-s3-policy.json