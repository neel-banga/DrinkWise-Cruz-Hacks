import requests
import random
import speech_recognition as sr
from pydub import AudioSegment

# Check if someones voice is slurring

# Get phrase to say
def get_phrase(num):
    words = 'https://www.mit.edu/~ecprice/wordlist.10000'
    response = requests.get(words)

    all_words_undecoded = response.content.splitlines()

    all_words = []

    for i in all_words_undecoded:
        all_words.append(i.decode('utf-8'))

    phrase = ''

    for i in range(num):

        rand_num = random.randrange(len(all_words))

        phrase += ' '
        phrase += all_words[rand_num]

    return phrase

# Convert speech to text
def speech_to_text(file_path):
    r = sr.Recognizer()

    audio_file = sr.AudioFile(file_path)
    with audio_file as source:
        audio_text = r.record(source)

    return r.recognize_google(audio_text, language='en-US')


# Implementation of the levenshtein distance algorythm to see the distance (insertions, deletions) between strings
def levenshtein_distance(s1, s2):
    m = len(s1) + 1
    n = len(s2) + 1
    dp = [[0] * n for i in range(m)]

    for i in range(m):
        dp[i][0] = i
    for j in range(n):
        dp[0][j] = j

    for i in range(1, m):
        for j in range(1, n):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j]) + 1

    return dp[m - 1][n - 1]

# I want a percentile though, so I'll create a function to do that
def similarity(s1, s2):
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len)


def check_slurring(file_path, example):

    text = speech_to_text(file_path)
    similar_percent = similarity(text, example)

    return 1-similar_percent

def get_wav_file(video_path):
    try:
        audio = AudioSegment.from_file(video_path, format='mov')
        audio.export('test.wav', format='wav')
    except:
        audio = AudioSegment.from_file(video_path, format='mp4')
        audio.export('test.wav', format='wav')

def get_words():
    with open('phrase.txt', 'r') as file:
        contents = file.read()
        x = contents
    return x

if __name__ =='main':
    toxic, toxicity = check_slurring('hi.wav', 'hello hello hello hello hello hello hello hello')
