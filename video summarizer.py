# Import necessary libraries
import moviepy.editor as mp  # For extracting audio from video
import speech_recognition as sr  # For converting speech to text

clip_duration = 150  # Duration for each extracted audio clip in seconds
collected_text = []  # List to store extracted text

# Load the video file
video = mp.VideoFileClip("path.mp4")  
total_duration = int(video.duration)  # Get the total duration of the video

# Loop through the video in chunks
for start_time in range(0, total_duration, clip_duration):  
    end_time = min(start_time + clip_duration, total_duration)  # Set end time for each segment
    clip = video.subclip(start_time, end_time)  # Extract the video segment

    # Extract audio from the video segment and save it as a WAV file
    audio = "audio.wav"  
    clip.audio.write_audiofile(audio)  

    # Initialize speech recognition
    r = sr.Recognizer()  
    with sr.AudioFile(audio) as source:  
        data = r.record(source)  # Read the audio data
          
        # Convert speech to text
        text = r.recognize_google(data)  
        print(text)  # Print extracted text
        collected_text.append(text)  # Append text to the list

# Combine all extracted text into a single string
input_text = " ".join(collected_text)  
print("The text is:", input_text)  

# Import transformers for text summarization
from transformers import BartTokenizer, BartForConditionalGeneration  

# Load the pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"  
tokenizer = BartTokenizer.from_pretrained(model_name)  
model = BartForConditionalGeneration.from_pretrained(model_name)  

# Count the number of words in the extracted text
numberWords = len(input_text.split())  
print("The number of words in the video is", numberWords)  

# Get user input for the summary ratio
ratio = float(input("Enter the ratio (in %) for the summary compared to the original text: "))  
minratio = ratio + (ratio / 40)  # Define minimum ratio
maxratio = ratio + (ratio / 15)  # Define maximum ratio
minnumber_of_words = int(minratio / 100 * numberWords)  # Calculate min word count for summary
maxnumber_of_words = int(maxratio / 100 * numberWords)  # Calculate max word count for summary
number_of_words = int(ratio / 100 * numberWords)  # Expected word count in summary
print("The number of words in the summary will most likely be:", number_of_words)  

# Tokenize input text and generate summary
inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)  
summary_ids = model.generate(inputs, max_length=maxnumber_of_words, min_length=minnumber_of_words, length_penalty=1, num_beams=4, early_stopping=True)  

# Decode the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)  
print("\nSummary:")  
print(summary)  

# Print the number of words in the summary
print("The number of words in the summary is:", len(summary.split()))
