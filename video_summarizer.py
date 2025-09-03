import moviepy.editor as mp
from whisper import load_model
from transformers import BartTokenizer, BartForConditionalGeneration  
import streamlit as st
import ffmpeg
import tempfile
import os
import yt_dlp

device = "cpu"

@st.cache_resource
def load_models():
    try:
        whisper_model = load_model('tiny').to(device)
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        return whisper_model, tokenizer, bart_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None
def downloadVideo(url:str, progress_callback = None):
    def progress_hook(d):
        if d['status']=='downloading' and progress_callback:
            percent = float(d.get('_percent_str', '0%').replace('%', ''))
            progress_callback(percent)
        elif d['status']=='finished'and progress_callback:
            progress_callback(100.0)

    ydl_opts= {
        "outtmpl":'./%(title)s.%(ext)s',
        "format":"worst[ext=mp4]/worst",
        "merge_output_format":"mp4",
        "process_hooks":[progress_hook]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url,download=True)
        return ydl.prepare_filename(info)
""def mkv2mp4(mkv_path,mp4_path):
    try:
        ffmpeg.input(mkv_path).output(mp4_path, c='copy').run()
    except:
        print("error ocurred")
""
def video_summarizer(video_path,ratio=30):
    video = mp.VideoFileClip(video_path)  
    total_duration = int(video.duration)
    whisper_model = load_model('tiny').to(device)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    #whisper_model,tokenizer,model = load_models()
    root = os.path.splitext(video_path)[0]
    audio_path = root + ".wav"
    audio = video.audio
    audio.write_audiofile(audio_path)
    transcribtion = whisper_model.transcribe(audio_path)
    os.remove(audio_path)
    text = transcribtion["text"]
    text_path = root + ".txt"

    numberWords = len(text.split())

    ratio = float(ratio)
    minratio = ratio + (ratio / 40)
    maxratio = ratio + (ratio / 15)
    minnumber_of_words = int(minratio / 100 * numberWords)
    maxnumber_of_words = int(maxratio / 100 * numberWords)
    number_of_words = int(ratio / 100 * numberWords)
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=maxnumber_of_words, min_length=minnumber_of_words, length_penalty=1, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary,number_of_words


st.title("Video Summarizer")
st.write("Upload a video file/ URL to get a summarized text of its content.")
videoType = st.selectbox("Select Video Type:", ["File", "URL"])

if videoType == "File":
    file_uploader = st.file_uploader("Video/s you want to summarize",type=["mp4","mov","mkv","avi"],accept_multiple_files=False)
    summary_percentage=st.slider("Percentage between the summary and full video text",0,80,30)
    button = st.button(label="**GENERATE SUMMARY**")
    st.markdown(""
        <style>
        div.stButton > button:first-child {
            background-color: #407a79; /* Green background */
            color: white; /* White text */
        }
        </style>
        "", unsafe_allow_html=True)
    if file_uploader and button:
        
        
        with tempfile.NamedTemporaryFile(delete=False,suffix=os.path.splitext(file_uploader.name)[1]) as tmp:
            tmp.write(file_uploader.read())
            tmp_path = tmp.name
        root , ext = os.path.splitext(tmp_path)
        if ext.lower() == ".mkv":
            pathmp4 = root + ".mp4"
            #mkv2mp4(tmp_path,pathmp4)
            tmp_path = pathmp4
        with st.spinner("summarizing the video"):
            summary , num_of_words = video_summarizer(tmp_path,summary_percentage)
            st.success("summary generated")
            st.text(f"Number of words in the summary:{number_of_words}")
            st.text_area("Your summary:",summary,height=300)
        os.remove(tmp_path)
    
elif videoType== "URL":
    url = st.text_input("Enter your URL")
    summary_percentage=st.slider("Percentage between the summary and full video text",0,80,30)
    button = st.button(label="**GENERATE SUMMARY**")
    st.markdown(""
        <style>
        div.stButton > button:first-child {
            background-color: #407a79; /* Green background */
            color: white; /* White text */
        }
        </style>
        "", unsafe_allow_html=True)
    if url and button:
        progressBar = st.progress(0)
        status = st.empty()

        def progress(percent):
            progressBar.progress(int(percent))
            status.text(f"Downloading..{percent:.2f}%")
        with st.spinner("Downloading the video.."):
            video_path = downloadVideo(url,progress_callback=progress)
            progressBar.progress(100)
            status.text("Download Completed")
            st.success(f"Downloaded to: {video_path}")
            st.video(video_path)
        with st.spinner("Summarizing the video"):
            summary,number_of_words = video_summarizer(video_path,summary_percentage)
            st.success("Summary generated")
            st.text(f"Number of words in the summary:{number_of_words}")
            st.text_area("Your summary:",summary,height=300)
        



