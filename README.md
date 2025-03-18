# Vital Curator

A web application that analyzes and compares patches based on their acoustic features. The engine uses feature extraction to help you shift a query to your desired sound.

## Features

- Compare patches based on multiple acoustic characteristics
- Adjust feature weights to prioritize specific sound qualities
- Interactive visualization of feature differences
- Real-time reranking of results based on user preferences
- Fast database for efficient similarity retrieval

## Technology

Built with Flask (Python) backend and vanilla JavaScript frontend. Uses R2/S3 storage for audio files and feature data.

## Setup

1. Clone repository
2. Copy `.env.example` to `.env` and configure storage credentials
3. Install dependencies with `pip install -r requirements.txt`
4. Run with `gunicorn main:app` or `python main.py` 