# Kubernetes Targets

These targets deploy Dev AI Lab to Kubernetes clusters using Kustomize for configuration management.

## Targets

### `make k8s-build`

Build and preview Kubernetes manifests.

```bash
make k8s-build
```

**With options:**

```bash
# Specific environment
make k8s-build KUSTOMIZE_OVERLAY=prod

# Specific cloud
make k8s-build KUSTOMIZE_OVERLAY=aws
```

**What it does:**
- Runs `kustomize build` on the specified overlay
- Outputs rendered YAML to stdout
- Does not apply anything to cluster

---

### `make k8s-apply`

Apply manifests to current Kubernetes context.

```bash
make k8s-apply
```

**With options:**

```bash
# Deploy dev environment
make k8s-apply KUSTOMIZE_OVERLAY=dev

# Deploy production
make k8s-apply KUSTOMIZE_OVERLAY=prod

# Deploy to AWS EKS
make k8s-apply KUSTOMIZE_OVERLAY=aws
```

**What it does:**
- Builds manifests with Kustomize
- Applies to current kubectl context
- Creates namespace, deployment, service, ingress, etc.

---

### `make k8s-delete`

Delete resources from cluster.

```bash
make k8s-delete
```

**With options:**

```bash
make k8s-delete KUSTOMIZE_OVERLAY=dev
```

---

## Overlays

### Environment Overlays

| Overlay | Replicas | CPU | Memory | Features |
|---------|----------|-----|--------|----------|
| `dev` | 1 | 0.5-1 | 1-2Gi | Minimal resources |
| `staging` | 1 | 1-2 | 2-4Gi | Production-like |
| `prod` | 2 | 2-4 | 4-8Gi | HPA, high availability |

### Cloud Overlays

| Overlay | Ingress | Storage | Identity |
|---------|---------|---------|----------|
| `aws` | ALB Ingress | EFS CSI | IRSA |
| `azure` | AGIC | Azure Files CSI | Workload Identity |
| `gcp` | GCE Ingress | Filestore CSI | Workload Identity |

---

## Configuration

### Base Resources

The base configuration in `deploy/kubernetes/base/` includes:

| Resource | File | Description |
|----------|------|-------------|
| Namespace | `namespace.yaml` | `devai` namespace |
| ConfigMap | `configmap.yaml` | Environment variables |
| Secret | `secret.yaml` | API keys (template) |
| ServiceAccount | `serviceaccount.yaml` | Pod identity |
| PVC | `pvc.yaml` | Persistent storage |
| Deployment | `deployment.yaml` | Pod specification |
| Service | `service.yaml` | ClusterIP service |
| Ingress | `ingress.yaml` | External access |

### Customizing Secrets

Before deploying, update the secrets:

```bash
# Copy and edit secret
cp deploy/kubernetes/base/secret.yaml deploy/kubernetes/overlays/dev/secret.yaml
```

Edit with actual values:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: devai-secrets
  namespace: devai
type: Opaque
stringData:
  OPENAI_API_KEY: "sk-..."
  ANTHROPIC_API_KEY: "sk-ant-..."
  GOOGLE_API_KEY: "..."
```

Add to kustomization:

```yaml
# deploy/kubernetes/overlays/dev/kustomization.yaml
resources:
  - ../../base
  - secret.yaml  # Add this
```

### Customizing Storage Class

For cloud deployments, update the storage class with your specific values:

**AWS (EFS):**
```yaml
# deploy/kubernetes/overlays/aws/storageclass.yaml
parameters:
  fileSystemId: fs-0123456789abcdef  # Your EFS ID
```

**Azure:**
```yaml
# deploy/kubernetes/overlays/azure/storageclass.yaml
parameters:
  skuName: Standard_LRS  # or Premium_LRS
```

### Customizing Service Account

For cloud IAM integration, update service account annotations:

**AWS (IRSA):**
```yaml
# deploy/kubernetes/overlays/aws/serviceaccount-patch.yaml
metadata:
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789:role/devai-role
```

**Azure (Workload Identity):**
```yaml
# deploy/kubernetes/overlays/azure/serviceaccount-patch.yaml
metadata:
  annotations:
    azure.workload.identity/client-id: <client-id>
```

**GCP (Workload Identity):**
```yaml
# deploy/kubernetes/overlays/gcp/serviceaccount-patch.yaml
metadata:
  annotations:
    iam.gke.io/gcp-service-account: devai@project.iam.gserviceaccount.com
```

---

## Common Workflows

### Deploy to Local Cluster (minikube/kind)

```bash
# Start local cluster
minikube start
# or
kind create cluster

# Deploy dev environment
make k8s-apply KUSTOMIZE_OVERLAY=dev

# Get service URL
minikube service devai -n devai --url
# or port-forward
kubectl port-forward -n devai svc/devai 8888:8888
```

### Deploy to AWS EKS

```bash
# 1. Configure kubectl for EKS
aws eks update-kubeconfig --name your-cluster --region us-east-1

# 2. Verify context
kubectl config current-context

# 3. Update AWS-specific configurations
vim deploy/kubernetes/overlays/aws/storageclass.yaml
vim deploy/kubernetes/overlays/aws/serviceaccount-patch.yaml

# 4. Deploy
make k8s-apply KUSTOMIZE_OVERLAY=aws

# 5. Check status
kubectl get pods -n devai
kubectl get ingress -n devai
```

### Deploy to Azure AKS

```bash
# 1. Configure kubectl for AKS
az aks get-credentials --resource-group myRG --name myAKS

# 2. Update Azure-specific configurations
vim deploy/kubernetes/overlays/azure/storageclass.yaml
vim deploy/kubernetes/overlays/azure/serviceaccount-patch.yaml

# 3. Deploy
make k8s-apply KUSTOMIZE_OVERLAY=azure

# 4. Check status
kubectl get pods -n devai
kubectl get ingress -n devai
```

### Deploy to GCP GKE

```bash
# 1. Configure kubectl for GKE
gcloud container clusters get-credentials my-cluster --region us-central1

# 2. Update GCP-specific configurations
vim deploy/kubernetes/overlays/gcp/storageclass.yaml
vim deploy/kubernetes/overlays/gcp/serviceaccount-patch.yaml

# 3. Deploy
make k8s-apply KUSTOMIZE_OVERLAY=gcp

# 4. Check status
kubectl get pods -n devai
kubectl get ingress -n devai
```

### Production Deployment

```bash
# 1. Review production settings
make k8s-build KUSTOMIZE_OVERLAY=prod

# 2. Apply with caution
make k8s-apply KUSTOMIZE_OVERLAY=prod

# 3. Verify HPA is active
kubectl get hpa -n devai
```

---

## Combining Overlays

For cloud + environment combinations, create composite overlays:

```bash
mkdir -p deploy/kubernetes/overlays/aws-prod
```

Create `deploy/kubernetes/overlays/aws-prod/kustomization.yaml`:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../prod

patches:
  - path: ../aws/ingress-patch.yaml
  - path: ../aws/serviceaccount-patch.yaml

resources:
  - ../aws/storageclass.yaml
```

Then deploy:

```bash
make k8s-apply KUSTOMIZE_OVERLAY=aws-prod
```

---

## Monitoring

### Check Pod Status

```bash
kubectl get pods -n devai -w
```

### View Logs

```bash
kubectl logs -n devai -l app=devai-lab -f
```

### Check Events

```bash
kubectl get events -n devai --sort-by='.lastTimestamp'
```

### Describe Resources

```bash
kubectl describe deployment devai -n devai
kubectl describe pod -n devai -l app=devai-lab
```

---

## Troubleshooting

### Pod Stuck in Pending

```bash
# Check events
kubectl describe pod -n devai -l app=devai-lab

# Common causes:
# - Insufficient resources
# - PVC not bound
# - Image pull errors
```

### PVC Not Bound

```bash
# Check PVC status
kubectl get pvc -n devai

# Check storage class
kubectl get storageclass

# For cloud storage, ensure CSI driver is installed
```

### Image Pull Errors

```bash
# Check image pull secret
kubectl get secrets -n devai

# For cloud registries, ensure service account has pull permissions
```

### Ingress Not Working

```bash
# Check ingress status
kubectl describe ingress devai -n devai

# Check ingress controller logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx

# For cloud ingress, check cloud-specific controller
```

### Service Not Accessible

```bash
# Port forward for testing
kubectl port-forward -n devai svc/devai 8888:8888

# Check endpoints
kubectl get endpoints devai -n devai
```
