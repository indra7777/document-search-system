#!/usr/bin/env python3
"""
Simple runner script for the document search system
Usage examples:
  python run_search.py --setup  # First time setup
  python run_search.py --web    # Start web interface
  python run_search.py --build /path/to/docs  # Build index
  python run_search.py --search "your query"  # Search
"""

import sys
import subprocess
import argparse
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def setup_models():
    """Setup instructions for downloading GGUF models"""
    print("\n🚀 Model Setup Instructions:")
    print("Download a GGUF model to the models/ directory:")
    print("\n📥 Option 1 - Phi-3 Mini (3.8B, recommended):")
    print("  wget -P models/ https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf")
    print("  mv models/Phi-3-mini-4k-instruct-q4.gguf models/llama-model.gguf")
    print("\n📥 Option 2 - Llama 3.1 8B (larger, more capable):")
    print("  wget -P models/ https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    print("  mv models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf models/llama-model.gguf")
    print("\n📥 Option 3 - Llama 3.2 1B (fastest, smallest):")
    print("  wget -P models/ https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    print("  mv models/Llama-3.2-1B-Instruct-Q4_K_M.gguf models/llama-model.gguf")

def run_web_interface():
    """Run the Streamlit web interface"""
    print("Starting web interface on all interfaces...")
    print("Access at: http://0.0.0.0:8502 or http://localhost:8502")
    print("External access: http://YOUR_SERVER_IP:8502")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "search_interface.py", 
        "--server.address", "0.0.0.0", 
        "--server.port", "8502",
        "--server.headless", "true"
    ])

def run_pipeline_command(args):
    """Run the main pipeline with given arguments"""
    cmd = [sys.executable, "main_pipeline.py"] + args
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Document Search System Runner")
    parser.add_argument('--setup', action='store_true', help='Setup the system')
    parser.add_argument('--web', action='store_true', help='Start web interface')
    parser.add_argument('--build', type=str, help='Build index from directory')
    parser.add_argument('--search', type=str, help='Search query')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--force-rebuild', action='store_true', help='Force rebuild')
    
    args = parser.parse_args()
    
    if args.setup:
        install_requirements()
        setup_models()
        print("\n✅ Setup complete! Now you can:")
        print("  - Build index: python run_search.py --build /path/to/docs")
        print("  - Start web UI: python run_search.py --web")
        print("  - Search: python run_search.py --search 'your query'")
        
    elif args.web:
        run_web_interface()
        
    elif args.build:
        pipeline_args = ['--build', args.build]
        if args.force_rebuild:
            pipeline_args.append('--force-rebuild')
        run_pipeline_command(pipeline_args)
        
    elif args.search:
        run_pipeline_command(['--search', args.search])
        
    elif args.interactive:
        run_pipeline_command(['--interactive'])
        
    else:
        print("🔍 Document Search System")
        print("\nQuick start:")
        print("  1. python run_search.py --setup")
        print("  2. python run_search.py --build /path/to/your/documents")
        print("  3. python run_search.py --web")
        print("\nOr use command line:")
        print("  python run_search.py --search 'your question'")
        print("  python run_search.py --interactive")

if __name__ == "__main__":
    main()