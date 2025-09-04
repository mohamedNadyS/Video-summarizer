import moviepy.editor as mp
import moviepy.config as mp_config
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  
import streamlit as st
import tempfile
import os
import yt_dlp
import gc
import subprocess
import shutil

def configure_ffmpeg():
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        try:
            mp_config.change_settings({"FFMPEG_BINARY": ffmpeg_path})
            return True
        except Exception as e:
            st.warning(f"Could not configure MoviePy with FFmpeg: {e}")
            return True
    else:
        st.error("FFmpeg not found. Please install FFmpeg to use this application.")
        return False

clip_duration = 60
collected_text = []
@st.cache_resource
def load_models():
    try:
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        bart_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
        return tokenizer, bart_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def downloadVideo(url: str, progress_callback=None):
    def progress_hook(d):
        if d['status'] == 'downloading' and progress_callback:
            percent_str = d.get('_percent_str', '0%').replace('%', '')
            try:
                percent = float(percent_str)
                progress_callback(percent)
            except ValueError:
                pass
        elif d['status'] == 'finished' and progress_callback:
            progress_callback(100.0)

    temp_dir = tempfile.mkdtemp()
    
    ydl_opts = {
        "outtmpl": os.path.join(temp_dir, '%(title)s.%(ext)s'),
        "format": "bestaudio+bestvideo[ext=mp4]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "progress_hooks": [progress_hook]
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")
        return None

def convert_to_wav_with_ffmpeg(input_path, output_path, start_time, duration):
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            st.warning(f"FFmpeg conversion failed: {result.stderr}")
            return False
        return True
        
    except subprocess.TimeoutExpired:
        st.warning(f"FFmpeg conversion timed out for segment {start_time}-{start_time+duration}")
        return False
    except Exception as e:
        st.warning(f"Error in FFmpeg conversion: {str(e)}")
        return False

def video_summarizer(video_path, ratio=30):
    global collected_text
    collected_text = []
    
    try:
        video = mp.VideoFileClip(video_path)
        total_duration = int(video.duration)
        video.close()
        
        st.info(f"Processing video of {total_duration} seconds duration...")
        progress_bar = st.progress(0)
        for i, start_time in enumerate(range(0, total_duration, clip_duration)):
            end_time = min(start_time + clip_duration, total_duration)
            chunk_duration = end_time - start_time
            progress = (i * clip_duration) / total_duration
            progress_bar.progress(min(progress, 1.0))
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                audio_path = tmp_audio.name
            if convert_to_wav_with_ffmpeg(video_path, audio_path, start_time, chunk_duration):
                r = sr.Recognizer()
                try:
                    with sr.AudioFile(audio_path) as source:
                        r.adjust_for_ambient_noise(source, duration=0.5)
                        data = r.record(source)
                        text = r.recognize_google(data)
                        collected_text.append(text)
                        st.write(f"Chunk {start_time}-{end_time}s: {len(text.split())} words transcribed")
                        
                except sr.UnknownValueError:
                    st.warning(f"Audio chunk {start_time}-{end_time}s was not understood, skipping.")
                except sr.RequestError as e:
                    st.error(f"Could not request results from Google Speech Recognition; {e}")
                except Exception as e:
                    st.warning(f"Error processing audio chunk {start_time}-{end_time}s: {str(e)}")

            try:
                os.remove(audio_path)
            except:
                pass
        
        progress_bar.progress(1.0)
        gc.collect()
        
        if not collected_text:
            st.error("No text could be extracted from the video. Please check if the video has audio.")
            return None, 0
        
        input_text = " ".join(collected_text)
        st.success(f"Transcribed {len(input_text.split())} total words from {len(collected_text)} chunks")
        tokenizer, model = load_models()
        if not tokenizer or not model:
            st.error("Could not load summarization models.")
            return None, 0
        
        numberWords = len(input_text.split())
        
        if numberWords < 10:
            st.warning("Not enough text extracted for meaningful summarization.")
            return input_text, numberWords
        
        ratio = float(ratio)
        minratio = max(5, ratio - (ratio / 10))
        maxratio = ratio + (ratio / 5)
        minnumber_of_words = max(10, int(minratio / 100 * numberWords))
        maxnumber_of_words = int(maxratio / 100 * numberWords)
        number_of_words = int(ratio / 100 * numberWords)
        
        max_input_length = 1000
        if len(input_text.split()) > max_input_length:
            words = input_text.split()
            chunks = [' '.join(words[i:i + max_input_length]) 
                     for i in range(0, len(words), max_input_length)]
            
            summaries = []
            for chunk in chunks:
                inputs = tokenizer.encode("summarize: " + chunk, 
                                        return_tensors="pt", 
                                        max_length=1024, 
                                        truncation=True)
                
                chunk_max_words = min(200, len(chunk.split()) // 3)
                summary_ids = model.generate(inputs, 
                                           max_length=chunk_max_words + 50,
                                           min_length=min(200, chunk_max_words),
                                           length_penalty=1.0,
                                           num_beams=4,
                                           early_stopping=True)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summaries.append(summary)

            combined_summary = " ".join(summaries)
            if len(combined_summary.split()) > number_of_words * 2:
                inputs = tokenizer.encode("summarize: " + combined_summary,
                                        return_tensors="pt",
                                        max_length=1024,
                                        truncation=True)
                summary_ids = model.generate(inputs,
                                           max_length=maxnumber_of_words + 50,
                                           min_length=minnumber_of_words,
                                           length_penalty=1.0,
                                           num_beams=4,
                                           early_stopping=True)
                final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            else:
                final_summary = combined_summary
        else:
            inputs = tokenizer.encode("summarize: " + input_text,
                                    return_tensors="pt",
                                    max_length=1024,
                                    truncation=True)
            summary_ids = model.generate(inputs,
                                       max_length=maxnumber_of_words + 50,
                                       min_length=minnumber_of_words,
                                       length_penalty=1.0,
                                       num_beams=4,
                                       early_stopping=True)
            final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        gc.collect()
        return final_summary, len(final_summary.split())
        
    except Exception as e:
        st.error(f"Error in video processing: {str(e)}")
        return None, 0

def main():
    if not configure_ffmpeg():
        return
    
    st.title("🎥 Video Summarizer")
    st.write("Upload a video file or provide a URL to get a summarized text of its content.")
    
    videoType = st.selectbox("Select Video Type:", ["File", "URL"])
    
    if videoType == "File":
        file_uploader = st.file_uploader(
            "Choose a video file to summarize",
            type=["mp4", "mov", "mkv", "avi", "webm"],
            accept_multiple_files=False
        )
        
        summary_percentage = st.slider(
            "Summary length (% of original transcript)",
            5, 50, 20,
            help="Lower percentages create more concise summaries"
        )
        
        button = st.button("**GENERATE SUMMARY**", type="primary")
        
        if file_uploader and button:
            with tempfile.NamedTemporaryFile(delete=False, 
                                           suffix=os.path.splitext(file_uploader.name)[1]) as tmp:
                tmp.write(file_uploader.read())
                tmp_path = tmp.name
            
            try:
                with st.spinner("Processing video..."):
                    summary, num_of_words = video_summarizer(tmp_path, summary_percentage)
                    
                if summary:
                    st.success("Summary generated successfully!")
                    st.metric("Summary length", f"{num_of_words} words")
                    st.text_area("Your summary:", summary, height=300)
                    st.download_button(
                        label="Download Summary",
                        data=summary,
                        file_name="video_summary.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("Failed to generate summary. Please try again with a different video.")
            
            finally:
                try:
                    os.remove(tmp_path)
                except:
                    pass
    
    elif videoType == "URL":
        url = st.text_input("Enter video URL (YouTube, etc.)")
        note = st.info("if 403 forbidden error happened try open source video that do NOT need cookies or log in like archive.org or download the youtube video using downr.org then upload it using file option")
        
        summary_percentage = st.slider(
            "Summary length (% of original transcript)",
            5, 50, 20,
            help="Lower percentages create more concise summaries"
        )
        
        button = st.button("**GENERATE SUMMARY**", type="primary")
        
        if url and button:
            progressBar = st.progress(0)
            status = st.empty()
            
            def progress(percent):
                progressBar.progress(int(min(percent, 100)))
                status.text(f"Downloading... {percent:.1f}%")
            
            try:
                with st.spinner("Downloading video..."):
                    video_path = downloadVideo(url, progress_callback=progress)
                    
                if video_path and os.path.exists(video_path):
                    progressBar.progress(100)
                    status.text("Download completed!")
                    st.success(f"Downloaded: {os.path.basename(video_path)}")
                    if st.checkbox("Show video preview"):
                        st.video(video_path)
                    
                    with st.spinner("Processing and summarizing video..."):
                        summary, number_of_words = video_summarizer(video_path, summary_percentage)
                        
                    if summary:
                        st.success("Summary generated successfully!")
                        st.metric("Summary length", f"{number_of_words} words")
                        st.text_area("Your summary:", summary, height=300)
                        st.download_button(
                            label="Download Summary",
                            data=summary,
                            file_name="video_summary.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("Failed to generate summary. Please try again.")
                else:
                    st.error("Failed to download video. Please check the URL and try again.")
                    
            finally:
                if 'video_path' in locals() and video_path and os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                    except:
                        pass

if __name__ == "__main__":
    main()