import streamlit as st
from pydub import AudioSegment
import numpy as np
import wave
import spacy
import speech_recognition as sr
from groq import Groq
from transformers import pipeline
import torch
import asyncio
from pydub import AudioSegment
import librosa
from concurrent.futures import ThreadPoolExecutor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

API_KEY = "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"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

segLen,frameRate,numMix = 3,50,128

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


def VoiceActivityDetection(wavData, frameRate):
    # uses the librosa library to compute short-term energy
    ste = librosa.feature.rms(y=wavData,hop_length=int(16000/frameRate)).T
    thresh = 0.1*(np.percentile(ste,97.5) + 9*np.percentile(ste,2.5))    # Trim 5% off and set threshold as 0.1x of the ste range
    return (ste>thresh).astype('bool')

def trainGMM(wavFile, frameRate, segLen, vad, numMix):
    wavData, _ = librosa.load(wavFile, sr=16000)
    mfcc = librosa.feature.mfcc(y=wavData, sr=16000, n_mfcc=20, hop_length=int(16000/frameRate)).T
    vad = np.reshape(vad, (len(vad),))
    if mfcc.shape[0] > vad.shape[0]:
        vad = np.hstack((vad, np.zeros(mfcc.shape[0] - vad.shape[0]).astype('bool'))).astype('bool')
    elif mfcc.shape[0] < vad.shape[0]:
        vad = vad[:mfcc.shape[0]]
    mfcc = mfcc[vad, :]

    print("Training GMM..")
    # GMM = GaussianMixture(n_components=numMix, covariance_type='diag').fit(mfcc)
    GMM = GaussianMixture(n_components=numMix, covariance_type='diag', reg_covar=1e-6, init_params='kmeans').fit(mfcc)
    segLikes = []
    segSize = frameRate * segLen
    for segI in range(int(np.ceil(float(mfcc.shape[0]) / (frameRate * segLen)))):
        startI = segI * segSize
        endI = (segI + 1) * segSize
        if endI > mfcc.shape[0]:
            endI = mfcc.shape[0] - 1
        if endI == startI:  # Reached the end of file
            break
        seg = mfcc[startI:endI, :]
        compLikes = np.sum(GMM.predict_proba(seg), 0)
        segLikes.append(compLikes / seg.shape[0])
    print("Training Done")

    return np.asarray(segLikes)

def SegmentFrame(clust, segLen, frameRate, numFrames):
    frameClust = np.zeros(numFrames)
    for clustI in range(len(clust)-1):
        frameClust[clustI*segLen*frameRate:(clustI+1)*segLen*frameRate] = clust[clustI]*np.ones(segLen*frameRate)
    frameClust[(clustI+1)*segLen*frameRate:] = clust[clustI+1]*np.ones(numFrames-(clustI+1)*segLen*frameRate)
    return frameClust

def speakerdiarisationdf(hyp, frameRate, wavFile):
    audioname=[]
    starttime=[]
    endtime=[]
    speakerlabel=[]
            
    spkrChangePoints = np.where(hyp[:-1] != hyp[1:])[0]
    if len(spkrChangePoints) > 0 and spkrChangePoints[0] != 0 and hyp[0] != -1:
        spkrChangePoints = np.concatenate(([0], spkrChangePoints))
    spkrLabels = [hyp[changePoint+1] for changePoint in spkrChangePoints]
    
    for spkrI, spkr in enumerate(spkrLabels[:-1]):
        if spkr != -1:
            audioname.append(wavFile.split('/')[-1])
            starttime.append((spkrChangePoints[spkrI] + 1) / float(frameRate))
            endtime.append(starttime[-1] + (spkrChangePoints[spkrI + 1] - spkrChangePoints[spkrI]) / float(frameRate))
            speakerlabel.append("Speaker " + str(int(spkr)))
    if spkrLabels[-1] != -1 and len(spkrChangePoints) > 0:
        audioname.append(wavFile.split('/')[-1])
        starttime.append(spkrChangePoints[-1] / float(frameRate))
        endtime.append((len(hyp) - spkrChangePoints[-1]) / float(frameRate))
        speakerlabel.append("Speaker " + str(int(spkrLabels[-1])))

    speakerdf = pd.DataFrame({"Audio": audioname, "StartTime": starttime, "EndTime": endtime, "SpeakerLabel": speakerlabel})
    return speakerdf

def perform_speaker_diarization(audio_path):
    wavData, _ = librosa.load(audio_path, sr=16000)
    vad = VoiceActivityDetection(wavData, frameRate)
    segLikes = trainGMM(audio_path, frameRate, segLen, vad, numMix)
    clustModel = AgglomerativeClustering(n_clusters=None, distance_threshold=0.2)
    clustLabels = clustModel.fit_predict(segLikes)
    hyp = SegmentFrame(clustLabels, segLen, frameRate, len(vad))
    speaker_df = speakerdiarisationdf(hyp, frameRate, audio_path)
    return speaker_df

def transcribe_audio_with_whisper(audio_path):
    transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-kannada-medium", chunk_length_s=30, device=device)
    transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="kn", task="transcribe")
    transcription = transcribe(audio_path)["text"]
    return transcription

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

def extract_audio_segment(audio_path, start_time, end_time, segment_path):
    """
    Extracts a segment from an audio file.
    """
    sound = AudioSegment.from_file(audio_path)
    start_time_ms = start_time * 1000
    end_time_ms = end_time * 1000
    extract = sound[start_time_ms:end_time_ms]
    extract.export(segment_path, format="wav")

async def async_transcribe_and_translate_segment(segment_path, executor):
    loop = asyncio.get_event_loop()
    # Run the synchronous transcription function in a separate thread
    transcription = await loop.run_in_executor(executor, transcribe_audio_with_whisper, segment_path)
    # Run the synchronous translation function in a separate thread
    translation = await loop.run_in_executor(executor, translate_text_kn_to_en, transcription)
    return transcription, translation

async def process_speaker_segments(audio_path, speaker_df):
    # Use a ThreadPoolExecutor to run synchronous functions without blocking the async loop
    with ThreadPoolExecutor() as executor:
        tasks = []
        for i, row in speaker_df.iterrows():
            segment_path = f"segment_{i}.wav"
            extract_audio_segment(audio_path, row['StartTime'], row['EndTime'], segment_path)
            # Schedule the transcription and translation as an asynchronous task
            task = async_transcribe_and_translate_segment(segment_path, executor)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        for result in results:
            transcription, translation = result
            # You can process the results here
            st.write(f'Transcription: {transcription}')
            st.write(f'Translation: {translation}')

# Streamlit app begins here
st.markdown("<h1 style='text-align: center; color: #12ABDB; pb:4'>Vocalyzer</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        audio_format = uploaded_file.name.split('.')[-1]
        audio_path = f"temp_audio.{audio_format}"
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
    
    if audio_format != "wav":
        # Convert to wav format using PyDub for compatibility with speech_recognition
        sound = AudioSegment.from_file(audio_path, format=audio_format)
        audio_path = "temp_audio_converted.wav"
        sound.export(audio_path, format="wav")
    
    st.divider()
    speaker_df = perform_speaker_diarization(audio_path)
    asyncio.run(process_speaker_segments(audio_path, speaker_df))
    

else:
    st.write("Please upload an audio file to get started.")
