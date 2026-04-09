# STREAMLIT CLOUD DEPLOYMENT SETUP SCRIPT (PowerShell)
# This script helps you deploy to both GitHub and Streamlit Cloud

Write-Host "===================================" -ForegroundColor Cyan
Write-Host "🚀 Legal BERT NLP - Deployment Setup" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: GitHub Setup
Write-Host "📋 STEP 1: GitHub Repository Setup" -ForegroundColor Yellow
Write-Host "====================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "You need to:"
Write-Host "1. Go to https://github.com/new"
Write-Host "2. Create a new repository called: legal-bert-nlp"
Write-Host "3. Choose public repository"
Write-Host "4. Click 'Create repository'"
Write-Host ""
Read-Host "Press Enter once you've created the repository"
Write-Host ""

# Step 2: Add GitHub Remote
Write-Host "📤 STEP 2: Connecting to GitHub" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow
Write-Host ""
$GITHUB_USER = Read-Host "Enter your GitHub username"
$REPO_URL = "https://github.com/$GITHUB_USER/legal-bert-nlp.git"

Write-Host "Adding remote: $REPO_URL"
git remote add origin $REPO_URL

# Step 3: Verify connection
Write-Host ""
Write-Host "Verifying connection..." -ForegroundColor Green
git remote -v
Write-Host ""

# Step 4: Push to GitHub
Write-Host "📤 STEP 3: Pushing Code to GitHub" -ForegroundColor Yellow
Write-Host "===================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Pushing code to GitHub..." -ForegroundColor Green
git branch -M main
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Successfully pushed to GitHub!" -ForegroundColor Green
    $GITHUB_URL = "https://github.com/$GITHUB_USER/legal-bert-nlp"
} else {
    Write-Host "❌ Push failed. Make sure:" -ForegroundColor Red
    Write-Host "  - GitHub credentials are correct"
    Write-Host "  - Repository exists at: $REPO_URL"
    exit 1
}

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "🎉 STEP 4: Enable GitHub Pages" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To enable GitHub Pages:"
Write-Host "1. Go to: https://github.com/$GITHUB_USER/legal-bert-nlp/settings"
Write-Host "2. Scroll to 'Pages' section"
Write-Host "3. Set source to 'Deploy from branch'"
Write-Host "4. Select: main branch → /docs folder"
Write-Host "5. Click 'Save'"
Write-Host ""
Write-Host "Your documentation will be at:"
Write-Host "  https://$GITHUB_USER.github.io/legal-bert-nlp/" -ForegroundColor Cyan
Write-Host ""

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "🎉 STEP 5: Deploy to Streamlit Cloud" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To deploy to Streamlit Cloud:" -ForegroundColor Green
Write-Host "1. Go to https://share.streamlit.io/"
Write-Host "2. Click 'Deploy an app'"
Write-Host "3. Select GitHub repository: $GITHUB_USER/legal-bert-nlp"
Write-Host "4. Set the main file path to: app/streamlit_app.py"
Write-Host "5. Click 'Deploy!'"
Write-Host ""
Write-Host "Your app will be at:"
Write-Host "  https://legal-bert-nlp.streamlit.app" -ForegroundColor Cyan
Write-Host ""

Write-Host "====================================" -ForegroundColor Green
Write-Host "✅ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host ""
Write-Host "📚 Your project is now deployed to:" -ForegroundColor Green
Write-Host "   - GitHub: $GITHUB_URL"
Write-Host "   - Documentation: https://$GITHUB_USER.github.io/legal-bert-nlp/"
Write-Host "   - Streamlit App: https://legal-bert-nlp.streamlit.app"
Write-Host ""
Write-Host "📖 Next steps:"
Write-Host "   1. Test the Streamlit Cloud app"
Write-Host "   2. Share the links with others"
Write-Host "   3. The app will auto-update when you push to GitHub"
Write-Host ""
