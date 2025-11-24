from flask import Flask, render_template, request, jsonify
from ibm_watson import NaturalLanguageUnderstandingV1, SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions, EmotionOptions, KeywordsOptions

app = Flask(__name__)

# -----------------------------
# IBM NLU
# -----------------------------
NLU_API_KEY = "eUcO0M_4jov53yc60E50eGEq4VSiY5VW13xMoZLRpRYu"
NLU_API_URL = "https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/81809b0d-2619-4f6f-829d-a4415f14b980"

nlu_auth = IAMAuthenticator(NLU_API_KEY)
nlu = NaturalLanguageUnderstandingV1(version='2021-08-01', authenticator=nlu_auth)
nlu.set_service_url(NLU_API_URL)

# -----------------------------
# IBM Speech-to-Text
# -----------------------------
STT_API_KEY = "BViO0Q1bGzEAFQW3v2W_3FSdzsugcO1V03SeHUQ6xagC"
STT_API_URL = "https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/1b520324-42cc-42ab-9c28-0d4767ef2379"

stt_auth = IAMAuthenticator(STT_API_KEY)
stt = SpeechToTextV1(authenticator=stt_auth)
stt.set_service_url(STT_API_URL)

# -----------------------------
# Home
# -----------------------------
@app.route('/')
def index():
    return render_template("index.html")


# -----------------------------
# Sentiment
# -----------------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']

    result = nlu.analyze(
        text=text,
        features=Features(sentiment=SentimentOptions())
    ).get_result()

    sentiment = result['sentiment']['document']['label'].capitalize()
    score = round(result['sentiment']['document']['score'], 2)

    return render_template("index.html",
                           text=text,
                           label="Sentiment",
                           result=sentiment,
                           score=score)


# -----------------------------
# Emotion
# -----------------------------
@app.route('/emotion', methods=['POST'])
def emotion():
    text = request.form['text']

    result = nlu.analyze(
        text=text,
        features=Features(emotion=EmotionOptions())
    ).get_result()

    emotions = result['emotion']['document']['emotion']
    top = max(emotions, key=emotions.get).capitalize()

    return render_template("index.html",
                           text=text,
                           label="Emotion",
                           result=top,
                           score=None)


# -----------------------------
# Language Detection
# -----------------------------
@app.route('/language', methods=['POST'])
def language():
    text = request.form['text']

    result = nlu.analyze(
        text=text,
        features=Features(keywords=KeywordsOptions(limit=1))
    ).get_result()

    lang = result['language'].capitalize()

    return render_template("index.html",
                           text=text,
                           label="Language",
                           result=lang,
                           score=None)


# -----------------------------
# Speech-to-Text (Mic)
# -----------------------------
@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    audio = request.files['audio']

    stt_result = stt.recognize(
        audio=audio,
        content_type='audio/webm',
        model='en-US_BroadbandModel'
    ).get_result()

    transcript = stt_result['results'][0]['alternatives'][0]['transcript']

    # auto-sentiment for alert box
    sentiment_result = nlu.analyze(
        text=transcript,
        features=Features(sentiment=SentimentOptions())
    ).get_result()

    sentiment = sentiment_result['sentiment']['document']['label'].capitalize()
    score = round(sentiment_result['sentiment']['document']['score'], 2)

    return jsonify({
        "transcript": transcript,
        "sentiment": sentiment,
        "score": score
    })


if __name__ == '__main__':
    app.run(debug=True)
