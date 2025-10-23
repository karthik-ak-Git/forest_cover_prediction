# 🚀 Production Deployment Guide

## Overview

This guide covers deploying the Forest Cover Prediction system to production using Docker, Kubernetes, and cloud providers.

---

## 📋 Prerequisites

### Required Tools
- Docker Desktop (latest)
- kubectl (Kubernetes CLI)
- Terraform (≥1.0)
- AWS CLI / Azure CLI (for cloud deployment)
- Helm (optional, for Kubernetes package management)

### Required Access
- Container registry access (GitHub Container Registry, Docker Hub, or AWS ECR)
- Kubernetes cluster (EKS, AKS, or GKE)
- Cloud provider account with appropriate permissions

---

## 🐳 Docker Deployment

### 1. Local Docker Deployment

```bash
# Clone repository
git clone https://github.com/karthik-ak-Git/forest_cover_prediction.git
cd forest_cover_prediction

# Build and run with Docker Compose
docker-compose up -d

# Check services
docker-compose ps

# View logs
docker-compose logs -f backend

# Access application
# Frontend: http://localhost
# API: http://localhost/api/docs
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
# MLflow: http://localhost:5000
```

### 2. Build Individual Images

```bash
# Build backend image
docker build -t forest-cover-backend:latest .

# Run backend
docker run -d \
  --name forest-backend \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host/db \
  -e REDIS_URL=redis://redis:6379 \
  forest-cover-backend:latest

# Test
curl http://localhost:8000/health
```

### 3. Push to Registry

```bash
# GitHub Container Registry
docker tag forest-cover-backend:latest ghcr.io/karthik-ak-git/forest_cover_prediction:latest
docker push ghcr.io/karthik-ak-git/forest_cover_prediction:latest

# Docker Hub
docker tag forest-cover-backend:latest your-username/forest-cover:latest
docker push your-username/forest-cover:latest
```

---

## ☸️ Kubernetes Deployment

### 1. Prepare Kubernetes Cluster

```bash
# For AWS EKS
eksctl create cluster \
  --name forest-cover-cluster \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.large \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 5 \
  --managed

# Configure kubectl
aws eks update-kubeconfig --name forest-cover-cluster --region us-east-1
```

### 2. Create Secrets

```bash
# Create namespace
kubectl create namespace forest-cover

# Create secrets
kubectl create secret generic forest-cover-secrets \
  --from-literal=database-url='postgresql://user:pass@host:5432/db' \
  --from-literal=secret-key='your-secret-key' \
  --from-literal=redis-url='redis://redis:6379' \
  -n forest-cover
```

### 3. Deploy Application

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/storage.yaml -n forest-cover
kubectl apply -f k8s/deployment.yaml -n forest-cover
kubectl apply -f k8s/ingress.yaml -n forest-cover

# Check deployment
kubectl get pods -n forest-cover
kubectl get services -n forest-cover
kubectl get ingress -n forest-cover

# View logs
kubectl logs -f deployment/forest-cover-backend -n forest-cover
```

### 4. Configure Ingress

```bash
# Install nginx-ingress controller
helm install nginx-ingress ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace

# Install cert-manager for SSL
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Get external IP
kubectl get service -n ingress-nginx
```

---

## ☁️ Cloud Deployment

### AWS Deployment with Terraform

```bash
cd terraform/

# Initialize Terraform
terraform init

# Plan deployment
terraform plan \
  -var="aws_region=us-east-1" \
  -var="environment=production" \
  -var="db_password=secure_password"

# Apply infrastructure
terraform apply \
  -var="aws_region=us-east-1" \
  -var="environment=production" \
  -var="db_password=secure_password"

# Get outputs
terraform output cluster_endpoint
terraform output rds_endpoint
terraform output redis_endpoint
```

### Azure Deployment

```bash
# Create resource group
az group create --name forest-cover-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group forest-cover-rg \
  --name forest-cover-aks \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials \
  --resource-group forest-cover-rg \
  --name forest-cover-aks
```

---

## 🔒 Security Configuration

### 1. SSL/TLS Certificate

```bash
# Using Let's Encrypt with cert-manager
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
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: forest-cover-network-policy
  namespace: forest-cover
spec:
  podSelector:
    matchLabels:
      app: forest-cover
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
EOF
```

---

## 📊 Monitoring Setup

### 1. Prometheus & Grafana

```bash
# Install Prometheus
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --create-namespace

# Install Grafana
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set adminPassword='admin'

# Get Grafana password
kubectl get secret --namespace monitoring grafana \
  -o jsonpath="{.data.admin-password}" | base64 --decode

# Port forward to access
kubectl port-forward -n monitoring svc/grafana 3000:80
```

### 2. Configure Dashboards

```bash
# Import predefined dashboard
# Navigate to Grafana UI → Import → Upload JSON
# Use dashboards from grafana/ directory
```

---

## 🔄 CI/CD Integration

### GitHub Actions

Already configured in `.github/workflows/ci-cd.yml`

**Triggers:**
- Push to `main` → Deploy to production
- Push to `develop` → Deploy to staging
- Pull request → Run tests only

**Required Secrets:**
- `GITHUB_TOKEN` (automatically provided)
- `AWS_ACCESS_KEY_ID` (if using AWS)
- `AWS_SECRET_ACCESS_KEY` (if using AWS)
- `KUBE_CONFIG` (Kubernetes config for deployment)

### Manual Trigger

```bash
# Via GitHub UI: Actions → CI/CD Pipeline → Run workflow

# Via CLI
gh workflow run ci-cd.yml
```

---

## 🧪 Testing Deployment

### Health Checks

```bash
# Check health endpoint
curl https://your-domain.com/health

# Expected response:
# {
#   "status": "healthy",
#   "timestamp": "2025-10-23T12:00:00",
#   "version": "2.0.0",
#   "redis": "connected"
# }
```

### Load Testing

```bash
# Install k6
# Then run load test
k6 run tests/performance/load-test.js

# Or use Apache Bench
ab -n 1000 -c 10 https://your-domain.com/api/predict
```

### Smoke Tests

```bash
# Get authentication token
TOKEN=$(curl -X POST "https://your-domain.com/token" \
  -d "username=demo&password=demo" | jq -r '.access_token')

# Make prediction
curl -X POST "https://your-domain.com/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @sample_input.json
```

---

## 📈 Scaling

### Horizontal Pod Autoscaling

Already configured in `k8s/deployment.yaml`

```bash
# Check HPA status
kubectl get hpa -n forest-cover

# Manual scaling
kubectl scale deployment forest-cover-backend \
  --replicas=5 \
  -n forest-cover
```

### Vertical Scaling

```bash
# Update resource limits
kubectl set resources deployment forest-cover-backend \
  --limits=cpu=2000m,memory=4Gi \
  --requests=cpu=500m,memory=1Gi \
  -n forest-cover
```

---

## 🔧 Troubleshooting

### View Logs

```bash
# All pods
kubectl logs -l app=forest-cover -n forest-cover --tail=100

# Specific pod
kubectl logs <pod-name> -n forest-cover -f

# Previous crashed container
kubectl logs <pod-name> -n forest-cover --previous
```

### Debug Pod

```bash
# Exec into pod
kubectl exec -it <pod-name> -n forest-cover -- /bin/bash

# Check environment variables
kubectl exec <pod-name> -n forest-cover -- env

# Check network
kubectl exec <pod-name> -n forest-cover -- curl http://redis:6379
```

### Common Issues

**Issue: Pods not starting**
```bash
kubectl describe pod <pod-name> -n forest-cover
kubectl get events -n forest-cover --sort-by='.lastTimestamp'
```

**Issue: Cannot connect to database**
```bash
# Test from pod
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- \
  psql postgresql://user:pass@host:5432/db
```

---

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)

---

## 🆘 Support

For issues:
1. Check logs: `kubectl logs -l app=forest-cover`
2. Review metrics: Grafana dashboard
3. Check health: `/health` endpoint
4. GitHub Issues: [Create an issue](https://github.com/karthik-ak-Git/forest_cover_prediction/issues)

---

**Last Updated:** October 2025  
**Version:** 2.0.0  
**Maintainer:** Karthik A K
