#!/bin/bash
# STREAMLIT CLOUD DEPLOYMENT SETUP SCRIPT
# This script helps you deploy to both GitHub and Streamlit Cloud

echo "==================================="
echo "🚀 Legal BERT NLP - Deployment Setup"
echo "==================================="
echo ""

# Step 1: GitHub Setup
echo "📋 STEP 1: GitHub Repository Setup"
echo "===================================="
echo ""
echo "You need to:"
echo "1. Go to https://github.com/new"
echo "2. Create a new repository called: legal-bert-nlp"
echo "3. Choose public repository"
echo "4. Click 'Create repository'"
echo ""
read -p "Press Enter once you've created the repository..."
echo ""

# Step 2: Add GitHub Remote
echo "📤 STEP 2: Connecting to GitHub"
echo "================================"
echo ""
read -p "Enter your GitHub username: " GITHUB_USER
REPO_URL="https://github.com/$GITHUB_USER/legal-bert-nlp.git"

echo "Adding remote: $REPO_URL"
git remote add origin $REPO_URL

# Step 3: Verify connection
echo ""
echo "Verifying connection..."
git remote -v
echo ""

# Step 4: Push to GitHub
echo "📤 STEP 3: Pushing Code to GitHub"
echo "==================================="
echo ""
echo "Pushing code to GitHub (might ask for credentials)..."
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo "✅ Successfully pushed to GitHub!"
    GITHUB_URL="https://github.com/$GITHUB_USER/legal-bert-nlp"
else
    echo "❌ Push failed. Make sure:"
    echo "  - GitHub credentials are correct"
    echo "  - Repository exists at: $REPO_URL"
    exit 1
fi

echo ""
echo "====================================="
echo "🎉 STEP 4: Enable GitHub Pages"
echo "====================================="
echo ""
echo "To enable GitHub Pages:"
echo "1. Go to: https://github.com/$GITHUB_USER/legal-bert-nlp/settings"
echo "2. Scroll to 'Pages' section"
echo "3. Set source to 'Deploy from branch'"
echo "4. Select: main branch → /docs folder"
echo "5. Click 'Save'"
echo ""
echo "Your documentation will be at:"
echo "  https://$GITHUB_USER.github.io/legal-bert-nlp/"
echo ""

echo "====================================="
echo "🎉 STEP 5: Deploy to Streamlit Cloud"
echo "====================================="
echo ""
echo "To deploy to Streamlit Cloud:"
echo "1. Go to https://share.streamlit.io/"
echo "2. Click 'Deploy an app'"
echo "3. Select GitHub repository: $GITHUB_USER/legal-bert-nlp"
echo "4. Set the main file path to: app/streamlit_app.py"
echo "5. Click 'Deploy!'"
echo ""
echo "Your app will be at:"
echo "  https://legal-bert-nlp.streamlit.app"
echo ""

echo "====================================="
echo "✅ SETUP COMPLETE!"
echo "====================================="
echo ""
echo "📚 Your project is now deployed to:"
echo "   - GitHub: $GITHUB_URL"
echo "   - Documentation: https://$GITHUB_USER.github.io/legal-bert-nlp/"
echo "   - Streamlit App: https://legal-bert-nlp.streamlit.app"
echo ""
echo "📖 Next steps:"
echo "   1. Test the Streamlit Cloud app"
echo "   2. Share the links with others"
echo "   3. The app will auto-update when you push to GitHub"
echo ""
