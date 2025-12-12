# 🔐 DharmaMind Secret Management Guide

This guide covers best practices for managing secrets in production.

## Quick Start (Development)

For local development, create a `.env` file:

```bash
cp .env.example .env
# Edit .env with your values
```

## Production Secret Management Options

### Option 1: AWS Secrets Manager (Recommended for AWS)

```python
# backend/app/config/secrets_aws.py
import boto3
import json
from functools import lru_cache

@lru_cache()
def get_secret(secret_name: str) -> dict:
    """Retrieve secret from AWS Secrets Manager."""
    client = boto3.client('secretsmanager', region_name='us-east-1')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Usage
secrets = get_secret("dharmamind/production")
JWT_SECRET_KEY = secrets["jwt_secret_key"]
DATABASE_URL = secrets["database_url"]
```

**Setup Steps:**
1. Create secret in AWS Console or CLI:
   ```bash
   aws secretsmanager create-secret \
     --name dharmamind/production \
     --secret-string '{"jwt_secret_key":"your-secret","database_url":"postgresql://..."}'
   ```

2. Grant IAM permissions to your service:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Action": ["secretsmanager:GetSecretValue"],
       "Resource": "arn:aws:secretsmanager:*:*:secret:dharmamind/*"
     }]
   }
   ```

### Option 2: HashiCorp Vault

```python
# backend/app/config/secrets_vault.py
import hvac

def get_vault_client():
    """Get authenticated Vault client."""
    client = hvac.Client(url='https://vault.example.com')
    client.auth.approle.login(
        role_id=os.getenv('VAULT_ROLE_ID'),
        secret_id=os.getenv('VAULT_SECRET_ID')
    )
    return client

def get_secret(path: str, key: str) -> str:
    """Retrieve secret from Vault."""
    client = get_vault_client()
    secret = client.secrets.kv.v2.read_secret_version(path=path)
    return secret['data']['data'][key]

# Usage
JWT_SECRET_KEY = get_secret("dharmamind/production", "jwt_secret_key")
```

**Setup Steps:**
1. Install Vault and initialize
2. Enable KV secrets engine:
   ```bash
   vault secrets enable -path=dharmamind kv-v2
   ```
3. Store secrets:
   ```bash
   vault kv put dharmamind/production \
     jwt_secret_key="your-secret" \
     database_url="postgresql://..."
   ```

### Option 3: Google Secret Manager

```python
# backend/app/config/secrets_gcp.py
from google.cloud import secretmanager

def get_secret(project_id: str, secret_id: str, version: str = "latest") -> str:
    """Retrieve secret from GCP Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Usage
JWT_SECRET_KEY = get_secret("dharmamind-prod", "jwt-secret-key")
```

### Option 4: Azure Key Vault

```python
# backend/app/config/secrets_azure.py
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

def get_secret(vault_url: str, secret_name: str) -> str:
    """Retrieve secret from Azure Key Vault."""
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=vault_url, credential=credential)
    return client.get_secret(secret_name).value

# Usage
JWT_SECRET_KEY = get_secret(
    "https://dharmamind-vault.vault.azure.net/",
    "jwt-secret-key"
)
```

## Environment-Based Configuration

```python
# backend/app/config/secrets.py
import os
from enum import Enum

class SecretProvider(str, Enum):
    ENV = "env"
    AWS = "aws"
    VAULT = "vault"
    GCP = "gcp"
    AZURE = "azure"

def get_secret(key: str, default: str = None) -> str:
    """
    Unified secret retrieval based on configured provider.
    """
    provider = os.getenv("SECRET_PROVIDER", SecretProvider.ENV)
    
    if provider == SecretProvider.ENV:
        return os.getenv(key, default)
    
    elif provider == SecretProvider.AWS:
        from .secrets_aws import get_secret as aws_get
        secrets = aws_get(os.getenv("AWS_SECRET_NAME", "dharmamind/production"))
        return secrets.get(key.lower(), default)
    
    elif provider == SecretProvider.VAULT:
        from .secrets_vault import get_secret as vault_get
        return vault_get(os.getenv("VAULT_PATH", "dharmamind/production"), key)
    
    elif provider == SecretProvider.GCP:
        from .secrets_gcp import get_secret as gcp_get
        return gcp_get(os.getenv("GCP_PROJECT"), key.lower().replace("_", "-"))
    
    elif provider == SecretProvider.AZURE:
        from .secrets_azure import get_secret as azure_get
        return azure_get(os.getenv("AZURE_VAULT_URL"), key.lower().replace("_", "-"))
    
    return default
```

## Required Secrets

| Secret Name | Description | Min Length |
|-------------|-------------|------------|
| `JWT_SECRET_KEY` | JWT signing key | 32 chars |
| `DATABASE_URL` | PostgreSQL connection | - |
| `REDIS_URL` | Redis connection | - |
| `STRIPE_SECRET_KEY` | Stripe API key | - |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook verification | - |
| `GOOGLE_CLIENT_SECRET` | Google OAuth secret | - |
| `SMTP_PASSWORD` | Email service password | - |

## Secret Rotation

### Automated Rotation (AWS)

```python
# backend/scripts/rotate_secrets.py
import boto3
import secrets

def rotate_jwt_secret():
    """Rotate JWT secret key with zero downtime."""
    client = boto3.client('secretsmanager')
    
    # Generate new secret
    new_secret = secrets.token_urlsafe(32)
    
    # Update with both old and new (for graceful rotation)
    current = client.get_secret_value(SecretId="dharmamind/production")
    current_data = json.loads(current['SecretString'])
    
    # Keep old key temporarily
    current_data['jwt_secret_key_old'] = current_data.get('jwt_secret_key')
    current_data['jwt_secret_key'] = new_secret
    
    client.update_secret(
        SecretId="dharmamind/production",
        SecretString=json.dumps(current_data)
    )
    
    print("✅ JWT secret rotated. Old key preserved for 24h grace period.")
```

### Rotation Schedule

| Secret | Rotation Frequency | Method |
|--------|-------------------|--------|
| JWT_SECRET_KEY | Every 90 days | Automated |
| DATABASE_URL | On breach/compromise | Manual |
| STRIPE_SECRET_KEY | On breach/compromise | Stripe Dashboard |
| SMTP_PASSWORD | Every 180 days | Email Provider |

## Kubernetes Secrets

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: dharmamind-secrets
type: Opaque
stringData:
  JWT_SECRET_KEY: "${JWT_SECRET_KEY}"
  DATABASE_URL: "${DATABASE_URL}"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dharmamind-backend
spec:
  template:
    spec:
      containers:
      - name: backend
        envFrom:
        - secretRef:
            name: dharmamind-secrets
```

## Docker Secrets (Swarm)

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  backend:
    image: dharmamind/backend:latest
    secrets:
      - jwt_secret
      - db_url
    environment:
      JWT_SECRET_KEY_FILE: /run/secrets/jwt_secret
      DATABASE_URL_FILE: /run/secrets/db_url

secrets:
  jwt_secret:
    external: true
  db_url:
    external: true
```

## Security Best Practices

1. **Never commit secrets** - Use `.gitignore`
2. **Rotate regularly** - At least every 90 days
3. **Use different secrets per environment** - Dev ≠ Staging ≠ Prod
4. **Audit access** - Log who accesses secrets
5. **Least privilege** - Only grant necessary access
6. **Encrypt at rest** - Use encrypted secret stores
7. **Monitor for leaks** - Use tools like git-secrets, truffleHog

## Generating Secure Secrets

```bash
# Generate 32-byte secret
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate with OpenSSL
openssl rand -base64 32

# Generate UUID-based
python -c "import uuid; print(str(uuid.uuid4()).replace('-', ''))"
```

## Validation

Run environment validation before deployment:

```bash
python backend/app/security/env_validator.py
```

This will check:
- All required secrets are present
- Secrets meet minimum length requirements
- No default/test values in production
- Proper format for specific secrets (e.g., Stripe keys)
