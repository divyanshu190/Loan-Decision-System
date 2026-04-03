#!/bin/bash

echo "Running training..."
python3 train.py --data cs-training.csv

echo "Running validation..."
python3 validate_model.py

echo "Starting app..."
python3 api/app.py
