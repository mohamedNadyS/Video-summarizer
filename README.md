# Video to Text and Summarization Script

## Overview

This project extracts audio from a video file, converts it to text using openAI whisper, and summarizes the extracted text using Facebook's BART model.
You have URL/file options to in the UI that created & deployed by streamlit works with mo4, mov, mkv, and avi files

## Features  

- Extracts audio from a given video file in chunks.  
- Converts the extracted speech into text using SpeechRecognition.  
- Uses facebook/bart-large-cnn for text summarization.  
- Allows users to specify the desired summary ratio.  
