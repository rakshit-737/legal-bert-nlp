# 🚀 Deployment Guide - Legal BERT NLP Streamlit App

## Quick Start: Streamlit Cloud (Recommended - Easiest & Free)

### 1. Push Code to GitHub
```bash
git init
git add .
git commit -m "Initial commit: Legal BERT NLP complete"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/legal-bert-nlp.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Click **"Deploy an app"**
3. Select your GitHub repo: `YOUR_USERNAME/legal-bert-nlp`
4. Main file path: `app/streamlit_app.py`
5. Click **"Deploy!"**

**Benefits:**
- ✅ Free tier available
- ✅ 1GB memory, CPU for free
- ✅ Custom domain support
- ✅ Auto-deploy on GitHub push
- ✅ Perfect for prototypes & demos

**Limitations:**
- ~1GB RAM (may struggle with large models)
- No GPU by default
- Restarts every hour

---

## Option 2: Docker + Cloud Run (Google Cloud)

### Create Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Deploy to Google Cloud Run
```bash
# Install Google Cloud CLI
# Then authenticate
gcloud auth login

# Build & deploy
gcloud run deploy legal-bert-nlp \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --timeout 3600
```

**Benefits:**
- ✅ Pay-per-request pricing
- ✅ Auto-scaling
- ✅ More resources available
- ✅ Docker containers

**Cost:** ~$0.00028/sec (~$24/month if always running)

---

## Option 3: AWS EC2 (Full Control)

### 1. Launch EC2 Instance
```bash
# Create Ubuntu 22.04 instance (t3.medium or larger)
# Open Security Group ports:
#   - Port 22 (SSH)
#   - Port 8501 (Streamlit)
```

### 2. Connect & Setup
```bash
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python & dependencies
sudo apt install python3.10 python3-pip git -y

# Clone your repo
git clone https://github.com/YOUR_USERNAME/legal-bert-nlp.git
cd legal-bert-nlp

# Install requirements
pip install -r requirements.txt

# Run with screen (keeps running after logout)
screen -S streamlit
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
# Press Ctrl+A, then D to detach
```

### 3. Setup Domain (Optional)
```bash
# Point domain DNS to your EC2 instance IP
# Then use Let's Encrypt for HTTPS
sudo apt install certbot -y
sudo certbot certonly --standalone -d yourdomain.com
```

**Cost:** $15-30/month (t3.medium)

---

## Option 4: Heroku (Deprecated but Still Works - Manual Setup)

⚠️ **Heroku free tier ended March 2023**, but you can use paid dynos.

```bash
# Install Heroku CLI, then:
heroku login
heroku create legal-bert-nlp
git push heroku main
```

**Cost:** $7-50/month depending on dyno size

---

## Option 5: Local Network (Home/Office)

### Run Locally
```bash
streamlit run app/streamlit_app.py
```

Access at: `http://localhost:8501`

### Share Over Network
```bash
streamlit run app/streamlit_app.py --server.address 0.0.0.0
```

Access from any machine on network: `http://YOUR_IP:8501`

### Use ngrok for Remote Access
```bash
# Install ngrok
pip install pyngrok

# Expose locally
streamlit run app/streamlit_app.py &
pyngrok.connect(8501)
```

Creates public URL to your local machine!

---

## Comparison Table

| Option | Cost | Setup Time | Performance | Best For |
|--------|------|-----------|-------------|----------|
| **Streamlit Cloud** | Free | 5 min | Basic | Demos, prototypes, learning |
| **Google Cloud Run** | Pay-per-use | 15 min | Good | Production, auto-scaling |
| **AWS EC2** | $15-30/mo | 30 min | Excellent | Full control, 24/7 |
| **Heroku** | $7-50/mo | 10 min | Good | Simple deployment |
| **Local + ngrok** | Free | 2 min | Variable | Testing, internal sharing |

---

## Recommended Setup for Production

### 💎 Best Overall: Google Cloud Run (Serverless)
1. ✅ Pay only for what you use
2. ✅ Auto-scales with traffic
3. ✅ Easy Docker deployment
4. ✅ Generous free tier ($300/month credit)

### For 24/7 Operations: AWS EC2
1. ✅ Predictable monthly cost
2. ✅ Full control & customization
3. ✅ Easy horizontal scaling
4. ✅ Large model support

### Quick Demo: Streamlit Cloud
1. ✅ Zero setup, just push to GitHub
2. ✅ Free tier perfect for portfolio/CV
3. ✅ Auto-deploys on every push

---

## Quick Start Commands

### 1. Streamlit Cloud (Fastest)
```bash
# Just push to GitHub
git push origin main
# Then deploy at https://share.streamlit.io/
```

### 2. Local + Share Link
```bash
pip install pyngrok
streamlit run app/streamlit_app.py
# Public URL generated automatically
```

### 3. Docker Local
```bash
docker build -t legal-bert-nlp .
docker run -p 8501:8501 legal-bert-nlp
# Visit http://localhost:8501
```

---

## Environment Variables (Optional)

Create `.streamlit/secrets.toml` for API keys or config:
```toml
[database]
host = "your_db_host"
port = 5432

[api]
key = "your_api_key"
```

Never commit secrets! Use cloud provider's secret manager instead.

---

## Performance Tips

1. **Model Caching**: Already enabled with `@st.cache_resource` ✅
2. **Image Optimization**: Compress PDFs before upload
3. **Memory**: Load models on startup, not per request ✅
4. **Batch Processing**: Use batch mode for 100+ documents
5. **Hardware**: Use GPU instances for faster inference

Select instance type based on:
- Small documents: CPU instance (cheaper)
- Large/frequent requests: GPU instance (faster)

---

## Monitoring & Debugging

### Log streaming
```bash
# On Streamlit Cloud dashboard - automatic

# On AWS EC2
ssh -i key.pem ubuntu@ip
tail -f /home/ubuntu/streamlit.log
```

### Performance monitoring
- Streamlit Cloud: Built-in dashboard
- Google Cloud Run: Cloud Monitoring
- AWS EC2: CloudWatch or Prometheus

---

## Summary

**I recommend: Streamlit Cloud for free tier, Google Cloud Run for scale**

Get started in 5 minutes:
1. Push code to GitHub
2. Go to https://share.streamlit.io/
3. Connect your repo
4. Done! ✅

Questions? Check [Streamlit Docs](https://docs.streamlit.io/deploy/streamlit-cloud)
