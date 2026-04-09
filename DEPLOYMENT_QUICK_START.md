# 🚀 Deploying Legal BERT NLP to Streamlit Cloud + GitHub

Your project is **100% ready to deploy**. This guide will walk you through the setup process.

## ⚡ Quick Start (5 minutes)

### Option 1: Interactive Setup (Recommended)

Run the interactive deployment assistant:

```bash
# On Windows PowerShell
python deploy.py

# On macOS/Linux
python3 deploy.py
```

This will guide you through all steps with visual feedback.

---

### Option 2: Manual Setup

## Step 1️⃣: Create GitHub Repository

1. Go to **https://github.com/new**
2. Create a new repository called: `legal-bert-nlp`
3. Choose **Public** visibility
4. **Important**: Do NOT initialize with README (we have one)
5. Click **"Create repository"**

## Step 2️⃣: Connect Your Local Repository

```bash
cd e:\legal-bert-nlp

# Add GitHub as remote
git remote add origin https://github.com/YOUR_USERNAME/legal-bert-nlp.git

# Verify
git remote -v
```

## Step 3️⃣: Push to GitHub

```bash
# Rename branch to main
git branch -M main

# Push code
git push -u origin main
```

**You may be prompted for GitHub credentials** - enter your username and [personal access token](https://github.com/settings/tokens) as password.

## Step 4️⃣: Enable GitHub Pages (Optional but Recommended)

1. Go to: **https://github.com/YOUR_USERNAME/legal-bert-nlp/settings**
2. Scroll to **"Pages"** section
3. Under "Build and deployment":
   - Source: **Deploy from branch**
   - Branch: **main**
   - Folder: **/docs**
4. Click **"Save"**

Your documentation will be published at:
```
https://YOUR_USERNAME.github.io/legal-bert-nlp/
```

## Step 5️⃣: Deploy to Streamlit Cloud ⭐

This is where your **live app** will run!

### Option A: Using Streamlit's Web Interface (Easiest)

1. Go to **https://share.streamlit.io/**
2. Click **"Deploy an app"** (sign in with GitHub first)
3. Fill in the deployment form:
   - **Repository**: `YOUR_USERNAME/legal-bert-nlp`
   - **Branch**: `main`
   - **Main file path**: `app/streamlit_app.py`
4. Click **"Deploy!"**
5. Wait 2-3 minutes for deployment to complete

Your live app URL:
```
https://legal-bert-nlp.streamlit.app
```

### Option B: Using Streamlit CLI (Alternative)

```bash
pip install streamlit

streamlit deploy app/streamlit_app.py
```

---

## ✅ Verification Checklist

After deployment, verify everything works:

- [ ] GitHub repository created at `https://github.com/YOUR_USERNAME/legal-bert-nlp`
- [ ] Code pushed to GitHub (visible on repository page)
- [ ] GitHub Pages enabled (Settings → Pages)
- [ ] Documentation accessible at `https://YOUR_USERNAME.github.io/legal-bert-nlp/`
- [ ] Streamlit Cloud deployment started at https://share.streamlit.io/
- [ ] App accessible at `https://legal-bert-nlp.streamlit.app` (after 2-3 min)

---

## 📊 Your Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LOCAL DEVELOPMENT                         │
│  (e:\legal-bert-nlp - your computer)                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    git push origin main
                           │
        ┌──────────────────┴────────────────────┐
        ▼                                       ▼
┌──────────────────────┐             ┌──────────────────────┐
│   GITHUB REPOSITORY  │             │ STREAMLIT CLOUD      │
│                      │             │                      │
│ Version Control      │             │ Live Web App         │
│ Code hosting         │             │ Real-time updates    │
│                      │             │ Auto-scaling         │
└──────────────────────┘             └──────────────────────┘
        │
        │ Deploy from branch
        │ /docs folder
        ▼
┌──────────────────────┐
│   GITHUB PAGES       │
│                      │
│ Documentation site   │
│ Static website       │
│ Auto-generated docs  │
└──────────────────────┘
```

---

## 🔄 Continuous Deployment (Auto-Update)

Once everything is set up, updates are **automatic**:

```bash
# Make changes locally
# Edit files, test, etc.

# Push to GitHub
git add .
git commit -m "Your changes"
git push origin main

# ✅ Streamlit Cloud automatically redeploys within 1-2 minutes!
# ✅ GitHub Pages docs update within seconds!
```

---

## 📞 Troubleshooting

### Issue: "git push" fails with authentication error

**Solution:**
```bash
# Create a personal access token at: https://github.com/settings/tokens
# Use it as your password when prompted

# Or set up SSH keys: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
```

### Issue: Streamlit Cloud deployment fails

**Check:**
1. Repository is public
2. `app/streamlit_app.py` exists
3. All dependencies in `requirements.txt`
4. No GPU-only code (Streamlit Cloud uses CPU)

### Issue: GitHub Pages not showing

**Solution:**
1. Wait 5 minutes (first deployment takes longer)
2. Check Settings → Pages → "Pages settings" 
3. Ensure branch is set to `main` and folder to `/docs`

### Issue: Model loading is slow on Streamlit Cloud

**This is normal** (1-2 minutes first load). It's because:
- First-time model download from HuggingFace (~500MB)
- CPU-based inference (no GPU on free tier)

**Improvements:**
- Model caching is enabled (`@st.cache_resource`)
- Subsequent runs are instant
- Upgrade to Streamlit Cloud paid tier for GPU

---

## 🎯 What's Next?

### After successful deployment:

1. **Share your app**:
   - "Check out my Legal BERT NLP app: https://legal-bert-nlp.streamlit.app"
   - "Documentation: https://YOUR_USERNAME.github.io/legal-bert-nlp/"

2. **Add to portfolio**:
   - Link GitHub repo to your resume
   - Show live working demo to employers

3. **Get custom domain** (optional):
   - Streamlit Cloud paid plans support custom domains
   - GitHub Pages supports custom domains

4. **Scale based on usage**:
   - Free tier great for portfolio/demos
   - Upgrade for production use or high traffic

---

## 📚 Additional Resources

- [Streamlit Cloud Docs](https://docs.streamlit.io/deploy/streamlit-cloud)
- [GitHub Pages Guide](https://pages.github.com/)
- [Git & GitHub for Beginners](https://guides.github.com/)

---

## 💡 Pro Tips

1. **Monitor your app**: Streamlit Cloud dashboard shows app health & usage
2. **Custom branding**: Add your name/logo in the Streamlit config
3. **Share secret metrics**: Streamlit cloud creates shareable links with metrics
4. **Upgrades**: Streamlit Cloud paid tier includes:
   - GPU support
   - More resources
   - Priority support
   - Custom domains

---

## 🎉 You're All Set!

Your Legal BERT NLP project is now ready for the world to see:

```
📱 Live App:       https://legal-bert-nlp.streamlit.app
📖 Documentation:  https://YOUR_USERNAME.github.io/legal-bert-nlp/
🐙 Git Repository: https://github.com/YOUR_USERNAME/legal-bert-nlp
```

**Questions?** Check the GETTING_STARTED.md or README.md in your project!
