#!/usr/bin/env python3
"""
Book Scraping Tool - Main Application Entry Point
================================================

This is the main entry point for the book scraping and analysis tool.
Run this file to start the Streamlit web application.

Usage:
    streamlit run app.py

Author: Book Scraper Project
Version: 1.0.0
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the Streamlit app
from src.ui.streamlit_app import main

if __name__ == "__main__":
    main()
