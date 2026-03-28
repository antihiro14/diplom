# Project Architecture

## Overview

This project is designed to analyze food from images using computer vision and machine learning techniques. At the current stage, the system has a modular structure and includes several core components.

## Main Modules

- `main.py` — entry point of the application;
- `src/dataset.py` — dataset handling module;
- `src/train.py` — model training module;
- `src/utils.py` — utility functions;
- `models/classifier.py` — model architecture.

## Component Interaction

The system works as follows:

1. Data is loaded from the dataset.
2. Images are preprocessed.
3. Data is passed to the model.
4. The model is trained.
5. Results can be used for further evaluation.

## Future Improvements

The system can be extended with:
- ingredient detection;
- nutrition calculation;
- mobile integration.