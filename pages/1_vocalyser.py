import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from pydub import AudioSegment
import io
from groq import Groq
from transformers import pipeline
import torch
import wave
from nltk.sentiment import SentimentIntensityAnalyzer
 
sia = SentimentIntensityAnalyzer()
API_KEY = "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"
 
def find_audio_clarity(audioFilename):
    audio = AudioSegment.from_file(audioFilename, format="wav")
    loudness = audio.dBFS
    high_threshold = -10
    low_threshold = -30
 
    if loudness > high_threshold:
        loudness_rate = "High loudness"
    elif loudness > low_threshold:
        loudness_rate = "Medium loudness"
    else:
        loudness_rate = "Low loudness"
 
    audio = wave.open(audioFilename, "r")
    signal = audio.readframes(-1)
    signal = np.frombuffer(signal, dtype="int16")
    intensity = np.sqrt(np.mean(np.square(signal)))
    audio.close()
 
    high_threshold = 10000
    low_threshold = 5000
 
    if intensity > high_threshold:
        intensity_level = "High intensity"
    elif intensity > low_threshold:
        intensity_level = "Medium intensity"
    else:
        intensity_level = "Low intensity"
 
    return loudness, loudness_rate, intensity, intensity_level
 
def analyze_sentiment(text):
    return sia.polarity_scores(text)
 
# Function to detect voice activity
def VoiceActivityDetection(wavData, frameRate):
    ste = librosa.feature.rms(y=wavData, hop_length=int(16000 / frameRate)).T
    thresh = 0.1 * (np.percentile(ste, 97.5) + 9 * np.percentile(ste, 2.5))
    return (ste > thresh).astype('bool')
 
# Function to train GMM
def trainGMM(wavData, frameRate, segLen, vad, numMix):
    mfcc = librosa.feature.mfcc(y=wavData, sr=16000, n_mfcc=20, hop_length=int(16000 / frameRate)).T
    vad = np.reshape(vad, (len(vad),))
    if mfcc.shape[0] > vad.shape[0]:
        vad = np.hstack((vad, np.zeros(mfcc.shape[0] - vad.shape[0]).astype('bool'))).astype('bool')
    elif mfcc.shape[0] < vad.shape[0]:
        vad = vad[:mfcc.shape[0]]
    mfcc = mfcc[vad, :]
    GMM = GaussianMixture(n_components=numMix, covariance_type='diag').fit(mfcc)
    segLikes = []
    segSize = frameRate * segLen
    for segI in range(int(np.ceil(float(mfcc.shape[0]) / (frameRate * segLen)))):
        startI = segI * segSize
        endI = (segI + 1) * segSize
        if endI > mfcc.shape[0]:
            endI = mfcc.shape[0] - 1
        if endI == startI:
            break
        seg = mfcc[startI:endI, :]
        compLikes = np.sum(GMM.predict_proba(seg), 0)
        segLikes.append(compLikes / seg.shape[0])
    return np.asarray(segLikes)
 
# Function to segment frames
def SegmentFrame(clust, segLen, frameRate, numFrames):
    frameClust = np.zeros(numFrames)
    for clustI in range(len(clust) - 1):
        frameClust[clustI * segLen * frameRate:(clustI + 1) * segLen * frameRate] = clust[clustI] * np.ones(segLen * frameRate)
    frameClust[(clustI + 1) * segLen * frameRate:] = clust[clustI + 1] * np.ones(numFrames - (clustI + 1) * segLen * frameRate)
    return frameClust
 
def speakerdiarisationdf(hyp, frameRate, wavFile):
    audioname=[]
    starttime=[]
    endtime=[]
    speakerlabel=[]
           
    spkrChangePoints = np.where(hyp[:-1] != hyp[1:])[0]
    if spkrChangePoints[0]!=0 and hyp[0]!=-1:
        spkrChangePoints = np.concatenate(([0],spkrChangePoints))
    spkrLabels = []    
    for spkrHomoSegI in range(len(spkrChangePoints)):
        spkrLabels.append(hyp[spkrChangePoints[spkrHomoSegI]+1])
    for spkrI,spkr in enumerate(spkrLabels[:-1]):
        if spkr!=-1:
            audioname.append(wavFile.split('/')[-1].split('.')[0]+".wav")
            starttime.append((spkrChangePoints[spkrI]+1)/float(frameRate))
            endtime.append((spkrChangePoints[spkrI+1]-spkrChangePoints[spkrI])/float(frameRate))
            speakerlabel.append("Speaker "+str(int(spkr)))
    if spkrLabels[-1]!=-1:
        audioname.append(wavFile.split('/')[-1].split('.')[0]+".wav")
        starttime.append(spkrChangePoints[-1]/float(frameRate))
        endtime.append((len(hyp) - spkrChangePoints[-1])/float(frameRate))
        speakerlabel.append("Speaker "+str(int(spkrLabels[-1])))
    #
    speakerdf=pd.DataFrame({"Audio":audioname,"starttime":starttime,"endtime":endtime,"speakerlabel":speakerlabel})
   
    spdatafinal=pd.DataFrame(columns=['Audio','SpeakerLabel','StartTime','EndTime'])
    i=0
    k=0
    j=0
    spfind=""
    stime=""
    etime=""
    for row in speakerdf.itertuples():
        if(i==0):
            spfind=row.speakerlabel
            stime=row.starttime
        else:
            if(spfind==row.speakerlabel):
                etime=row.starttime        
            else:
                spdatafinal.loc[k]=[wavFile.split('/')[-1].split('.')[0]+".wav",spfind,stime,row.starttime]
                k=k+1
                spfind=row.speakerlabel
                stime=row.starttime
        i=i+1
    spdatafinal.loc[k]=[wavFile.split('/')[-1].split('.')[0]+".wav",spfind,stime,etime]
    return spdatafinal
 
def translate_text_kn_to_en(input_text):
    client = Groq(api_key=API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a Translator. You translate Kannada text to English text."
            },
            {
                "role": "user",
                "content": f"Translate the given text: {input_text}",
            }
        ],
        model='mixtral-8x7b-32768',
        temperature=0.1,
    )
    translated_text = chat_completion.choices[0].message.content
    return translated_text
 
def main():
    st.markdown("<h1 style='text-align: center; color: #12ABDB; pb:4'>Vocalyzer</h1>", unsafe_allow_html=True)
 
    uploaded_file = st.file_uploader("Upload an audio file", type="wav")
 
    if uploaded_file is not None:
        # Temporary save uploaded file to process
        audio_file_path = "temp_uploaded_file.wav"
        with open(audio_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
       
        try:
            loudness, loudness_rate, intensity, intensity_level = find_audio_clarity("temp_audio.wav")
           
            col1, col2 = st.columns(2)
           
            with col1:
                st.warning("Loudness")
                st.write(f"**dBFS**: {loudness}")
                if loudness_rate == "High loudness":
                    st.markdown("<h2 style='color: red;'>🔊 High</h2>", unsafe_allow_html=True)
                elif loudness_rate == "Medium loudness":
                    st.markdown("<h2 style='color: orange;'>🔉 Medium</h2>", unsafe_allow_html=True)
                else:
                    st.markdown("<h2 style='color: green;'>🔈 Low</h2>", unsafe_allow_html=True)
           
            with col2:
                st.warning("Intensity")
                st.write(f"**Value**: {intensity}")
                if intensity_level == "High intensity":
                    st.markdown("<h2 style='color: red;'>⚡ High</h2>", unsafe_allow_html=True)
                elif intensity_level == "Medium intensity":
                    st.markdown("<h2 style='color: orange;'>⚡ Medium</h2>", unsafe_allow_html=True)
                else:
                    st.markdown("<h2 style='color: green;'>⚡ Low</h2>", unsafe_allow_html=True)
 
        except Exception as e:
            st.error(f"An error occurred: {e}")
 
        st.divider()
 
        with st.spinner('Processing...'):
            # Load your input .wav audio file
            full_audio = AudioSegment.from_wav(audio_file_path)
 
            # Segment the audio file if necessary or just proceed to transcription
            # Assuming you're processing the entire file:
            start_time = 0  # start of the file
            end_time = len(full_audio)  # end of the file
 
            # Extract the segment; if you're processing the whole file, this step might not be necessary
            extracted_segment = full_audio[start_time:end_time]
 
            # Save the extracted segment; adjust this if you're working with multiple segments
            segment_path = "extracted_segment.wav"
            extracted_segment.export(segment_path, format="wav")
 
            # Transcribe the audio file
            device = "cuda" if torch.cuda.is_available() else "cpu"
            transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-kannada-medium", chunk_length_s=30, device=device)
            # Make sure to adjust 'chunk_length_s' and 'model' based on your requirements
 
            transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="kn", task="transcribe")
            transcription = transcribe(segment_path)["text"]
            st.write("Transcription:", transcription)
 
            # Translate the transcription
            translated_text = translate_text_kn_to_en(transcription)
            st.write("Translation:", translated_text)
 
            st.write("Analyzing Sentiment...")
            # Analyzing Sentiment with VADER
            sentiment_scores = analyze_sentiment(translated_text)
            st.write("Sentiment Scores:", sentiment_scores)
 
            # Using the 'compound' score to determine overall sentiment
            compound_score = sentiment_scores['compound']
            if compound_score > 0:
                st.markdown("### Overall Sentiment: Positive 🙂")
            elif compound_score < 0:
                st.markdown("### Overall Sentiment: Negative 😞")
            else:
                st.markdown("### Overall Sentiment: Neutral 😐")
 
        st.success("Processing complete!")
 
 
if __name__ == "__main__":
    main()