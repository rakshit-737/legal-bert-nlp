# 🚀 Complete Deployment Guide

Your Legal BERT NLP project is now enhanced with a premium UI/UX design and ready for production deployment. This guide covers all deployment options.

---

## 📊 What's Been Completed

✅ **Premium Streamlit UI**
- Advanced CSS design system with animations
- Multi-format document upload (PDF, DOCX, TXT, CSV, MD)
- Enhanced task modules with professional styling
- Improved error handling and user guidance

✅ **GitHub Pages Documentation**
- Modern landing page with hero section
- Feature showcase with hover animations
- Deployment option comparison
- Technology stack showcase

✅ **Code Optimization**
- Complete implementations (DomainSpecificAttention, clustering, error handling)
- Model caching for performance
- Batch processing support
- Multi-format file extraction

---

## 🌐 GitHub Pages Deployment (2 minutes)

Enable GitHub Pages to host your documentation at `https://rakshit-737.github.io/legal-bert-nlp`

### Step 1: Enable GitHub Pages
1. Go to: https://github.com/rakshit-737/legal-bert-nlp/settings
2. Scroll down to **"Pages"** section
3. Under **"Build and deployment"**:
   - **Source**: Select "Deploy from branch"
   - **Branch**: Select `main` and `/docs` folder
4. Click **Save**

### Step 2: Verify Deployment
- Wait 1-2 minutes for GitHub to process
- Your site will be available at: **https://rakshit-737.github.io/legal-bert-nlp**
- Check the "Pages" section for the live URL

### Step 3 (Optional): Custom Domain
If you have a custom domain:
1. In Pages settings, add your custom domain under "Custom domain"
2. Update your domain's DNS records to point to GitHub Pages

---

## 🎯 Streamlit Cloud Deployment (5 minutes)

Deploy your interactive app to the public internet at `https://legal-bert-nlp.streamlit.app`

### Prerequisites
- Streamlit Cloud account (free): https://share.streamlit.io
- Your GitHub repository is public

### Step 1: Connect GitHub to Streamlit
1. Visit: https://share.streamlit.io
2. Click **"New app"**
3. Select **"GitHub repository"**
4. Authorize Streamlit to access your GitHub account (1-time setup)

### Step 2: Configure Your App
1. **Repository**: Select `rakshit-737/legal-bert-nlp`
2. **Branch**: `main`
3. **Main file path**: `app/streamlit_app.py`
4. Click **"Deploy"**

### Step 3: Wait for Deployment
- Streamlit will build your app (2-3 minutes)
- Once complete, your app is live at: **https://legal-bert-nlp.streamlit.app**
- Streamlit will show logs during the build process

### Step 4: Manage Secrets (Optional)
If you need environment variables:
1. In Streamlit Cloud dashboard, access your app settings
2. Add secrets in the **"Secrets"** section
3. Format: Add to `.streamlit/secrets.toml`

---

## 🐳 Docker Deployment (Advanced)

For complete control over the deployment environment.

### Build Docker Image
```bash
docker build -t legal-bert-nlp .
```

### Run Locally
```bash
docker run -p 8501:8501 legal-bert-nlp
```

### Deploy to Cloud Run (Google Cloud)
```bash
# Requires: gcloud CLI configured and Google Cloud account

# Build for Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/legal-bert-nlp

# Deploy
gcloud run deploy legal-bert-nlp \
  --image gcr.io/PROJECT_ID/legal-bert-nlp \
  --memory 2Gb \
  --timeout 3600
```

---

## ☁️ AWS EC2 Deployment (Advanced)

For maximum control and GPU support.

### Launch EC2 Instance
1. EC2 Console → Launch Instance
2. **AMI**: Ubuntu 22.04 LTS
3. **Instance Type**: `t3.medium` (CPU) or `g4dn.xlarge` (GPU)
4. **Storage**: 30GB (SSD recommended)
5. **Security Group**: Allow ports 22 (SSH), 8501 (Streamlit)

### Connect & Install
```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update and install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv -y

# Clone your repository
git clone https://github.com/rakshit-737/legal-bert-nlp.git
cd legal-bert-nlp

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run Streamlit
streamlit run app/streamlit_app.py
```

### Access Your App
- Visit: `http://your-instance-ip:8501`

### Make It Production-Ready (Optional)
```bash
# Use systemd to manage Streamlit as a service
sudo tee /etc/systemd/system/streamlit.service << EOF
[Unit]
Description=Streamlit Legal BERT NLP
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/legal-bert-nlp
Environment="PATH=/home/ubuntu/legal-bert-nlp/venv/bin"
ExecStart=/home/ubuntu/legal-bert-nlp/venv/bin/streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable streamlit
sudo systemctl start streamlit
```

---

## 📱 Local Development Setup

To develop and test locally:

```bash
# Clone repository
git clone https://github.com/rakshit-737/legal-bert-nlp.git
cd legal-bert-nlp

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/streamlit_app.py

# App opens at: http://localhost:8501
```

---

## 🔒 Security Best Practices

### Before Deploying to Production

1. **Secrets Management**
   - Never commit API keys or credentials
   - Use `.streamlit/secrets.toml` for local development
   - Use platform-specific secret management (Streamlit Cloud, GitHub Secrets, AWS Secrets Manager)

2. **Environment Variables**
   - Copy `.streamlit/secrets.toml.template` to `.streamlit/secrets.toml`
   - Fill with your local development values
   - Never commit `secrets.toml` (it's in `.gitignore`)

3. **Input Validation**
   - The app validates file formats (PDF, DOCX, TXT, CSV, MD only)
   - File size limits are enforced by Streamlit

4. **Model Security**
   - Models are cached locally to prevent re-download
   - Use model checkpoints from trusted sources

---

## 📊 Deployment Comparison

| Feature | Streamlit Cloud | GitHub Pages | Docker | AWS EC2 |
|---------|-----------------|--------------|--------|---------|
| **Cost** | Free (basic) | Free | Free (image) | Pay per hour |
| **Setup Time** | 5 minutes | 2 minutes | 15 minutes | 30 minutes |
| **Maintenance** | None | None | Minimal | Manual |
| **Performance** | Good (shared) | N/A (static) | Custom | Excellent |
| **GPU Support** | No | N/A | Yes | Yes (premium) |
| **Custom Domain** | Yes (paid) | Yes (free) | Yes | Yes |
| **Recommended For** | Production | Documentation | High traffic | Enterprise |

---

## 🔄 Continuous Integration/Deployment

### Automatic Deployment on Push (GitHub Actions)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Trigger Streamlit Cloud Deploy
        run: |
          curl -X POST https://api.streamlit.cloud/v1/apps/build \
            -H "Authorization: Bearer ${{ secrets.STREAMLIT_TOKEN }}" \
            -d '{"repository": "rakshit-737/legal-bert-nlp"}'
```

---

## 📞 Monitoring & Troubleshooting

### Streamlit Cloud Monitoring
- Dashboard: https://share.streamlit.io → Your App → Settings
- Logs: Available in app settings
- Analytics: View traffic and usage metrics

### Common Issues & Solutions

**Issue**: App loads slowly
- ✅ **Solution**: Models are cached; first load takes 1-2 minutes
- Clear browser cache if persists

**Issue**: File upload fails
- ✅ Check file format (PDF, DOCX, TXT, CSV, MD only)
- ✅ Maximum file size: Depends on platform (2GB+ on Streamlit Cloud)

**Issue**: Out of Memory
- ✅ Reduce batch size in batch processing
- ✅ Process files individually instead of combining
- ✅ Use GPU instance for better performance (AWS EC2)

**Issue**: GitHub Pages not updating
- ✅ Wait 1-2 minutes after push
- ✅ Check Actions tab for build errors
- ✅ Verify `docs/` folder exists and `_config.yml` is valid

---

## ✅ Deployment Checklist

- [ ] GitHub Pages enabled (Settings → Pages)
- [ ] Streamlit Cloud deployed (share.streamlit.io)
- [ ] Both URLs are working
- [ ] File upload working (test with sample PDF/DOCX)
- [ ] Models loading successfully
- [ ] Classification/NER/Similarity features working
- [ ] GitHub Pages landing page displaying
- [ ] Custom styling applied (premium UI visible)

---

## 🎉 You're All Set!

Your Legal BERT NLP application is now:
- ✅ Live on Streamlit Cloud
- ✅ Documented on GitHub Pages
- ✅ Source code safe on GitHub
- ✅ Ready for production use

### Quick Links
- **Live App**: https://legal-bert-nlp.streamlit.app
- **Documentation**: https://rakshit-737.github.io/legal-bert-nlp
- **GitHub Repository**: https://github.com/rakshit-737/legal-bert-nlp
- **Issues/Support**: https://github.com/rakshit-737/legal-bert-nlp/issues

---

## 📚 Additional Resources

- **Streamlit Documentation**: https://docs.streamlit.io
- **GitHub Pages Guide**: https://pages.github.com
- **Docker Guide**: https://docs.docker.com
- **AWS EC2 Guide**: https://docs.aws.amazon.com/ec2/

---

**Built with ❤️ for legal professionals and researchers**
