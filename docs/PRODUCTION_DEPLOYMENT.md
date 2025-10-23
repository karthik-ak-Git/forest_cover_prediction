# 🚀 Production Deployment Guide

Complete guide for deploying the Forest Cover Prediction system to production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Docker Deployment](#local-docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Monitoring & Observability](#monitoring--observability)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Security Best Practices](#security-best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

- **Docker** (v20.10+)
- **Docker Compose** (v2.0+)
- **kubectl** (v1.24+)
- **Helm** (v3.0+) - Optional but recommended
- **Terraform** (v1.0+) - For cloud infrastructure
- **Git**
- **Python** (3.8+)

### Cloud Accounts (Choose one)

- AWS Account with EKS access
- Azure Account with AKS access
- GCP Account with GKE access

---

## Local Docker Deployment

### Step 1: Build Docker Image

```bash
# Clone repository
git clone https://github.com/karthik-ak-Git/forest_cover_prediction.git
cd forest_cover_prediction

# Build image
docker build -t forest-cover-api:latest .

# Verify image
docker images | grep forest-cover-api
```

### Step 2: Run with Docker Compose

```bash
# Start all services (API, Redis, Prometheus, Grafana)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Step 3: Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs

# Make test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "elevation": 2596,
    "aspect": 51,
    "slope": 3,
    "horizontal_distance_to_hydrology": 258,
    "vertical_distance_to_hydrology": 0,
    "horizontal_distance_to_roadways": 510,
    "hillshade_9am": 221,
    "hillshade_noon": 232,
    "hillshade_3pm": 148,
    "horizontal_distance_to_fire_points": 6279,
    "wilderness_area_0": 1,
    "wilderness_area_1": 0,
    "wilderness_area_2": 0,
    "wilderness_area_3": 0
  }'
```

---

## Kubernetes Deployment

### Step 1: Prepare Kubernetes Cluster

#### Using Minikube (Local)

```bash
# Start minikube
minikube start --cpus=4 --memory=8192

# Enable addons
minikube addons enable ingress
minikube addons enable metrics-server
```

#### Using Cloud (AWS EKS Example)

```bash
# Create cluster
eksctl create cluster \
  --name forest-cover-prod \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 4 \
  --managed

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name forest-cover-prod
```

### Step 2: Create Namespace

```bash
kubectl create namespace forest-cover
kubectl config set-context --current --namespace=forest-cover
```

### Step 3: Deploy Application

```bash
# Apply storage
kubectl apply -f k8s/storage.yaml

# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Apply ingress
kubectl apply -f k8s/ingress.yaml

# Verify deployment
kubectl get pods
kubectl get services
kubectl get ingress
```

### Step 4: Configure Secrets

```bash
# Create secret for API keys
kubectl create secret generic api-secrets \
  --from-literal=SECRET_KEY=your-secret-key \
  --from-literal=DB_PASSWORD=your-db-password

# Create secret for container registry
kubectl create secret docker-registry regcred \
  --docker-server=ghcr.io \
  --docker-username=your-username \
  --docker-password=your-token \
  --docker-email=your-email
```

### Step 5: Setup Horizontal Pod Autoscaler

```bash
# Create HPA
kubectl autoscale deployment forest-cover-api \
  --cpu-percent=70 \
  --min=2 \
  --max=10

# Verify HPA
kubectl get hpa
```

---

## Cloud Deployment

### AWS Deployment

#### Using Terraform

```bash
cd terraform

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var="environment=production"

# Apply configuration
terraform apply -var="environment=production"

# Get outputs
terraform output
```

#### Manual Setup

1. **Create ECR Repository**
```bash
aws ecr create-repository --repository-name forest-cover-api
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com
```

2. **Push Image**
```bash
docker tag forest-cover-api:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/forest-cover-api:latest
docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/forest-cover-api:latest
```

3. **Deploy to ECS/EKS**
   - Use AWS Console or CLI to create ECS service
   - Or follow Kubernetes deployment steps above for EKS

### Azure Deployment

```bash
# Create resource group
az group create --name forest-cover-rg --location eastus

# Create container registry
az acr create --resource-group forest-cover-rg \
  --name forestcoveracr --sku Basic

# Login to ACR
az acr login --name forestcoveracr

# Push image
docker tag forest-cover-api:latest forestcoveracr.azurecr.io/forest-cover-api:latest
docker push forestcoveracr.azurecr.io/forest-cover-api:latest

# Create AKS cluster
az aks create \
  --resource-group forest-cover-rg \
  --name forest-cover-aks \
  --node-count 3 \
  --enable-managed-identity \
  --attach-acr forestcoveracr

# Get credentials
az aks get-credentials --resource-group forest-cover-rg --name forest-cover-aks
```

### GCP Deployment

```bash
# Set project
gcloud config set project your-project-id

# Create GKE cluster
gcloud container clusters create forest-cover-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-2

# Get credentials
gcloud container clusters get-credentials forest-cover-cluster --zone us-central1-a

# Build and push to GCR
gcloud builds submit --tag gcr.io/your-project-id/forest-cover-api

# Deploy
kubectl apply -f k8s/
```

---

## Monitoring & Observability

### Prometheus Setup

```bash
# Add Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --create-namespace

# Port forward
kubectl port-forward -n monitoring svc/prometheus-server 9090:80
```

### Grafana Setup

```bash
# Install Grafana
helm install grafana grafana/grafana \
  --namespace monitoring

# Get admin password
kubectl get secret --namespace monitoring grafana -o jsonpath="{.data.admin-password}" | base64 --decode

# Port forward
kubectl port-forward -n monitoring svc/grafana 3000:80
```

### Application Logs

```bash
# View real-time logs
kubectl logs -f deployment/forest-cover-api

# View logs from all pods
kubectl logs -l app=forest-cover-api --tail=100

# Stream logs to local file
kubectl logs -f deployment/forest-cover-api > app.log
```

### Metrics Endpoints

- **Health**: `http://your-domain/health`
- **Metrics**: `http://your-domain/metrics`
- **Prometheus**: `http://your-domain:9090`
- **Grafana**: `http://your-domain:3000`

---

## CI/CD Pipeline

### GitHub Actions (Already Configured)

The repository includes a comprehensive CI/CD pipeline at `.github/workflows/ci-cd.yml`:

- **On Push to `develop`**: Deploy to staging
- **On Push to `main`**: Deploy to production
- **On PR**: Run tests and validation

### Manual Deployment

```bash
# Tag release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# This triggers production deployment via GitHub Actions
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/forest-cover-api

# Rollback to specific revision
kubectl rollout undo deployment/forest-cover-api --to-revision=2

# Check rollout status
kubectl rollout status deployment/forest-cover-api
```

---

## Security Best Practices

### 1. Enable HTTPS/TLS

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create certificate issuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### 2. Network Policies

```bash
# Apply network policy
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-network-policy
spec:
  podSelector:
    matchLabels:
      app: forest-cover-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          role: frontend
    ports:
    - protocol: TCP
      port: 8000
EOF
```

### 3. Security Scanning

```bash
# Scan Docker image
trivy image forest-cover-api:latest

# Scan Kubernetes manifests
kubesec scan k8s/deployment.yaml
```

### 4. Secrets Management

Use external secret managers:

```bash
# AWS Secrets Manager integration
kubectl create secret generic api-secrets \
  --from-literal=secret-key=$(aws secretsmanager get-secret-value \
    --secret-id forest-cover-secret \
    --query SecretString --output text)
```

---

## Troubleshooting

### Common Issues

#### 1. Pod Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name>

# Check events
kubectl get events --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> --previous
```

#### 2. Service Not Accessible

```bash
# Test service internally
kubectl run test-pod --rm -it --image=busybox -- wget -O- http://forest-cover-api:8000/health

# Check endpoints
kubectl get endpoints forest-cover-api
```

#### 3. High Memory/CPU Usage

```bash
# Check resource usage
kubectl top pods

# Describe pod resources
kubectl describe pod <pod-name> | grep -A 5 Requests

# Scale up
kubectl scale deployment forest-cover-api --replicas=5
```

#### 4. Database Connection Issues

```bash
# Test connectivity
kubectl run test-db --rm -it --image=postgres:13 -- psql -h <db-host> -U <user>

# Check secrets
kubectl get secret api-secrets -o yaml
```

### Performance Tuning

```yaml
# Update resource limits
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

### Health Checks

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

---

## Production Checklist

- [ ] SSL/TLS certificates configured
- [ ] Secrets properly managed (not in code)
- [ ] Resource limits set on all containers
- [ ] Autoscaling configured (HPA)
- [ ] Monitoring and alerting setup
- [ ] Backup strategy implemented
- [ ] Disaster recovery plan documented
- [ ] Security scanning in CI/CD
- [ ] Network policies applied
- [ ] Logging aggregation configured
- [ ] Performance baseline established
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Runbooks created for common issues

---

## Support

For deployment issues:
- GitHub Issues: https://github.com/karthik-ak-Git/forest_cover_prediction/issues
- Documentation: See `docs/` folder
- Email: karthik@example.com

---

**Last Updated**: October 2025
