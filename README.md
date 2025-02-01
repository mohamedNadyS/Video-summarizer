# Video to Text and Summarization Script

## Overview

This project extracts audio from a video file, converts it to text using Google's Speech Recognition API, and summarizes the extracted text using Facebook's BART model.  

## Features  

- Extracts audio from a given video file in chunks.  
- Converts the extracted speech into text using SpeechRecognition.  
- Uses facebook/bart-large-cnn for text summarization.  
- Allows users to specify the desired summary ratio.  

## Installation  

Before running the script, install the required dependencies:  

`bash
pip install SpeechRecognition moviepy transformers torch
