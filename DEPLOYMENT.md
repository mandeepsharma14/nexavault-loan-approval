# NexaVault Loan Approval System — Deployment Guide
## Copyright (c) 2026 Mandeep Sharma. All rights reserved.

---

## Deployment Options Overview

| Method | Best For | Complexity | Cost |
|--------|----------|------------|------|
| Local (Flask dev) | Development & testing | Low | Free |
| Docker (local) | Demo & staging | Low | Free |
| Docker + Nginx | Production on single VM | Medium | ~$20/mo |
| AWS EC2 + ECR | Scalable cloud | Medium | ~$50/mo |
| AWS ECS/Fargate | Auto-scaling | High | Pay-per-use |
| Render / Railway | Fastest cloud deploy | Low | Free–$25/mo |
| Kubernetes (EKS/GKE) | Enterprise-grade | High | $100+/mo |

---

## Option 1 — Local Development

```bash
python loan_approval_system.py      # train
cd app && python app.py             # serve at http://localhost:5000
```

---

## Option 2 — Docker (Local / Staging)

```bash
docker build -t nexavault-app .
docker run -d -p 5000:5000 \
  -e MODEL_PATH=app/nexavault_model.pkl \
  -e APP_ENV=production \
  --name nexavault-loan-api \
  nexavault-app

# Verify
curl http://localhost:5000/health
```

---

## Option 3 — Docker Compose + Nginx (Recommended Production)

### nginx.conf
```nginx
upstream nexavault {
    server nexavault-api:5000;
}
server {
    listen 80;
    server_name yourdomain.com;
    location / {
        proxy_pass http://nexavault;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 60s;
    }
}
```

```bash
docker-compose up --build -d
docker-compose ps        # check status
docker-compose logs -f   # live logs
```

---

## Option 4 — AWS EC2 Deployment

### Step 1: Launch EC2 Instance
- AMI: Ubuntu 22.04 LTS
- Instance type: t3.medium (2 vCPU, 4 GB RAM)
- Security group: open port 22 (SSH), 80 (HTTP), 5000 (API)

### Step 2: Install Docker on EC2
```bash
ssh -i key.pem ubuntu@<EC2_IP>
sudo apt update && sudo apt install -y docker.io docker-compose git
sudo usermod -aG docker ubuntu
newgrp docker
```

### Step 3: Deploy
```bash
git clone https://github.com/mandeep-sharma/nexavault-loan-approval.git
cd nexavault-loan-approval
cp .env.example .env && nano .env    # edit config
docker-compose up --build -d
```

### Step 4: SSL with Let's Encrypt
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

---

## Option 5 — AWS ECS (Fargate) — Auto-Scaling

### Build & Push to ECR
```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Build, tag, push
docker build -t nexavault-app .
docker tag nexavault-app:latest \
  <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/nexavault:latest
docker push \
  <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/nexavault:latest
```

### ECS Task Definition (key settings)
```json
{
  "family": "nexavault-task",
  "cpu": "512",
  "memory": "1024",
  "networkMode": "awsvpc",
  "containerDefinitions": [{
    "name": "nexavault-api",
    "image": "<ECR_URI>",
    "portMappings": [{"containerPort": 5000, "protocol": "tcp"}],
    "environment": [
      {"name": "APP_ENV", "value": "production"},
      {"name": "MODEL_PATH", "value": "app/nexavault_model.pkl"}
    ]
  }]
}
```

### Fargate Service + ALB
1. Create ECS Cluster (Fargate)
2. Create Task Definition with above config
3. Create Service with desired count = 2
4. Attach Application Load Balancer on port 80 → target port 5000
5. Enable Auto Scaling: CPU > 70% → scale out

---

## Option 6 — Render (Easiest Cloud Deploy)

```bash
# 1. Push code to GitHub
git push origin main

# 2. Go to render.com → New Web Service → Connect GitHub repo

# 3. Settings:
#    Build Command:  pip install -r requirements.txt && python loan_approval_system.py
#    Start Command:  gunicorn -w 2 -b 0.0.0.0:10000 app.app:app
#    Environment:    MODEL_PATH=app/nexavault_model.pkl

# 4. Deploy → get URL like https://nexavault-api.onrender.com
```

---

## Option 7 — Kubernetes (Enterprise)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexavault-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nexavault
  template:
    metadata:
      labels:
        app: nexavault
    spec:
      containers:
      - name: api
        image: nexavault-app:latest
        ports:
        - containerPort: 5000
        env:
        - name: MODEL_PATH
          value: "app/nexavault_model.pkl"
        - name: APP_ENV
          value: "production"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: nexavault-service
spec:
  selector:
    app: nexavault
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods
kubectl get services
```

---

## CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/deploy.yml
name: NexaVault Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with: {python-version: "3.11"}
    - run: pip install -r requirements.txt
    - run: python loan_approval_system.py
    - run: pytest tests/ -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build & push Docker image
      run: |
        docker build -t nexavault-app .
        # Push to your registry here
    - name: Deploy to server
      run: |
        ssh user@server "cd /app && git pull && docker-compose up --build -d"
```

---

## Production Checklist

- [ ] Set `DEBUG=false` in `.env`
- [ ] Use gunicorn (not Flask dev server)
- [ ] Enable HTTPS / SSL
- [ ] Set `SECRET_KEY` to a random 32-char string
- [ ] Configure API key authentication
- [ ] Enable rate limiting (Flask-Limiter)
- [ ] Set up logging to file / CloudWatch
- [ ] Configure health check endpoint
- [ ] Enable auto-restart on crash
- [ ] Set up model retraining schedule
- [ ] Back up model artifacts to S3 / GCS
- [ ] Monitor model drift over time

---

*Copyright (c) 2026 Mandeep Sharma. All rights reserved.*
*NexaVault Financial Corp — Intelligent Loan Approval System*
