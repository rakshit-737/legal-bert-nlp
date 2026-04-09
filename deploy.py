#!/usr/bin/env python3
"""
Interactive Streamlit Cloud Deployment Assistant
Guides you through deploying Legal BERT NLP to GitHub and Streamlit Cloud
"""

import subprocess
import sys
import webbrowser
from pathlib import Path

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*50}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*50}{Colors.END}\n")

def print_step(num, text):
    print(f"{Colors.BOLD}STEP {num}: {text}{Colors.END}")
    print("-" * 50)

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

def run_command(cmd, show_output=True):
    """Run a shell command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=not show_output)
        return result.returncode == 0
    except Exception as e:
        print_error(f"Command failed: {e}")
        return False

def main():
    print_header("🚀 Legal BERT NLP - Streamlit Cloud Deployment")
    
    print_info("This assistant will help you deploy your Legal BERT NLP app to:")
    print_info("  • GitHub (version control)")
    print_info("  • GitHub Pages (documentation)")
    print_info("  • Streamlit Cloud (live app)")
    
    # Check git
    print("\n" + Colors.YELLOW + "Checking prerequisites..." + Colors.END)
    if not run_command("git --version", show_output=False):
        print_error("Git not installed. Please install Git first.")
        sys.exit(1)
    print_success("Git is installed")
    
    # Step 1: GitHub Setup
    print_step(1, "Create GitHub Repository")
    print("1. Go to: https://github.com/new")
    print("2. Create a new repository called: legal-bert-nlp")
    print("3. Choose 'Public' visibility")
    print("4. Click 'Create repository'")
    input(f"\n{Colors.BOLD}Press Enter when ready...{Colors.END}")
    
    # Step 2: Configure Git Remote
    print_step(2, "Configure Git Remote")
    github_user = input(f"{Colors.BOLD}Enter your GitHub username: {Colors.END}").strip()
    
    if not github_user:
        print_error("GitHub username is required")
        sys.exit(1)
    
    repo_url = f"https://github.com/{github_user}/legal-bert-nlp.git"
    print_info(f"Repository URL: {repo_url}")
    
    # Add remote
    if run_command(f"git remote add origin {repo_url}", show_output=False):
        print_success("Git remote configured")
    else:
        print_error("Failed to add git remote")
        sys.exit(1)
    
    # Step 3: Push to GitHub
    print_step(3, "Push Code to GitHub")
    print_info("Running: git branch -M main")
    print_info("Running: git push -u origin main")
    print_info("(You may be asked for GitHub credentials)")
    
    run_command("git branch -M main")
    if run_command("git push -u origin main"):
        print_success("Code pushed to GitHub!")
    else:
        print_error("Failed to push to GitHub. Check credentials and try again.")
        sys.exit(1)
    
    # Step 4: Enable GitHub Pages
    print_step(4, "Enable GitHub Pages")
    github_url = f"https://github.com/{github_user}/legal-bert-nlp/settings"
    print_info("GitHub Pages needs manual setup:")
    print("\n1. Go to: " + Colors.CYAN + github_url + Colors.END)
    print("2. Scroll down to 'Pages' section")
    print("3. Under 'Build and deployment':")
    print("   - Source: Deploy from branch")
    print("   - Branch: main / /docs folder")
    print("4. Click 'Save'")
    
    open_settings = input(f"\n{Colors.BOLD}Open settings in browser? (y/n): {Colors.END}").strip().lower() == 'y'
    if open_settings:
        webbrowser.open(github_url)
    
    print_success("GitHub Pages setup instructions provided")
    
    # Step 5: Streamlit Cloud
    print_step(5, "Deploy to Streamlit Cloud")
    print_info("Streamlit Cloud deployment:")
    print("\n1. Go to: https://share.streamlit.io/")
    print("2. Click 'Deploy an app' (or sign in first)")
    print("3. Select your GitHub repo:")
    print_info(f"   Repository: {github_user}/legal-bert-nlp")
    print_info("   Branch: main")
    print_info("   Main file: app/streamlit_app.py")
    print("4. Click 'Deploy!'")
    print("5. Wait for deployment (2-3 minutes)")
    
    open_streamlit = input(f"\n{Colors.BOLD}Open Streamlit Cloud in browser? (y/n): {Colors.END}").strip().lower() == 'y'
    if open_streamlit:
        webbrowser.open("https://share.streamlit.io/")
    
    # Summary
    print_header("✅ Deployment Setup Complete!")
    
    print(f"{Colors.GREEN}Your project is now configured for deployment:{Colors.END}\n")
    print(f"📚 GitHub Repository:")
    print(f"   {Colors.CYAN}https://github.com/{github_user}/legal-bert-nlp{Colors.END}\n")
    print(f"📖 GitHub Pages (Documentation):")
    print(f"   {Colors.CYAN}https://{github_user}.github.io/legal-bert-nlp/{Colors.END}\n")
    print(f"🚀 Streamlit Cloud (Live App):")
    print(f"   {Colors.CYAN}https://legal-bert-nlp.streamlit.app{Colors.END}\n")
    
    print(f"{Colors.BOLD}Next steps:{Colors.END}")
    print("1. Finalize GitHub Pages settings (if not auto-enabled)")
    print("2. Wait for Streamlit Cloud to complete deployment")
    print("3. Test the app at: https://legal-bert-nlp.streamlit.app")
    print("4. Share the links with others!")
    print(f"\n{Colors.GREEN}Auto-updates: Any git push to main will redeploy automatically!{Colors.END}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Deployment setup cancelled.{Colors.END}\n")
        sys.exit(0)
