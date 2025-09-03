
"""
def create_gui():
    ctk.set_appearance_mode("dark")  
    ctk.set_default_color_theme("blue")  

    root = ctk.CTk()  
    root.title("Video Summarizer")  
    root.geometry("400x200")  
    global ratio_entry
    global video_path_entry
    video_path_label = ctk.CTkLabel(root, text="Enter Video Path:")  
    video_path_label.grid(pady=10)

    video_path_entry = ctk.CTkEntry(root, width=300)  
    video_path_entry.grid(pady=5)

    summarize_button = ctk.CTkButton(root, text="Summarize Video", command=run_video_summarizer)  
    summarize_button.grid(pady=20)
    the_summaryframe = ctk.CTkScrollableFrame(root, width=300,height=200)
    the_summaryframe.grid(pady=10,padx=10,fill="both",expand=True)
    summary = ctk.CTkLabel(the_summaryframe,text="summary will appers here when it is ready",wraplength=250)
    summary.grid(pady=10,padx=10)

    root.mainloop()
"""

"""for start_time in range(0, total_duration, clip_duration):  
        end_time = min(start_time + clip_duration, total_duration)
        clip = video.subclip(start_time, end_time)
        audio = "audio.wav"  
        clip.audio.write_audiofile(audio)  

        r = sr.Recognizer()  
        with sr.AudioFile(audio) as source:
            data = r.record(source)
            text = r.recognize_google(data)
            print(text)
            collected_text.append(text)"""