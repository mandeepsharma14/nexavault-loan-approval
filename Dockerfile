# NexaVault Financial Corp — Intelligent Loan Approval System
# Copyright (c) 2026 Mandeep Sharma. All rights reserved.

FROM python:3.11-slim

LABEL maintainer="Mandeep Sharma"
LABEL description="NexaVault Intelligent Loan Approval System"
LABEL version="2.0.0"

WORKDIR /nexavault

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Train the model at build time
RUN python loan_approval_system.py

EXPOSE 5000

ENV MODEL_PATH=app/nexavault_model.pkl
ENV APP_ENV=production
ENV DEBUG=false

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "app/app.py"]
