#!/bin/bash
# Start backend
uvicorn backend.main:app --reload --port 8000 &
# Start frontend
streamlit run frontend/app.py --server.port 8501
