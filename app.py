from flask import Flask, render_template, request, jsonify
from ibm_watson import NaturalLanguageUnderstandingV1, SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions

app = Flask(__name__)

# -----------------------------
# IBM NLU (Sentiment Analysis)
# -----------------------------
NLU_API_KEY = "eUcO0M_4jov53yc60E50eGEq4VSiY5VW13xMoZLRpRYu"
NLU_API_URL = "https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/81809b0d-2619-4f6f-829d-a4415f14b980"

nlu_auth = IAMAuthenticator(NLU_API_KEY)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=nlu_auth
)
nlu.set_service_url(NLU_API_URL)

# -----------------------------
# IBM STT (Speech-to-Text)
# -----------------------------
STT_API_KEY = "BViO0Q1bGzEAFQW3v2W_3FSdzsugcO1V03SeHUQ6xagC"
STT_API_URL = "https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/1b520324-42cc-42ab-9c28-0d4767ef2379"

stt_auth = IAMAuthenticator(STT_API_KEY)
stt = SpeechToTextV1(authenticator=stt_auth)
stt.set_service_url(STT_API_URL)

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

# Sentiment analysis route
@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text', '').strip()
    if not text or len(text) < 5:
        return render_template(
            'index.html',
            text=text,
            sentiment="⚠️ Please enter a longer sentence.",
            score="–"
        )
    try:
        response = nlu.analyze(
            text=text,
            features=Features(sentiment=SentimentOptions())
        ).get_result()

        sentiment = response['sentiment']['document']['label'].capitalize()
        score = round(response['sentiment']['document']['score'], 2)

        return render_template(
            'index.html',
            text=text,
            sentiment=sentiment,
            score=score
        )
    except Exception as e:
        return render_template(
            'index.html',
            text=text,
            sentiment=f"❌ Error: {str(e)}",
            score="–"
        )

# Speech-to-Text endpoint
@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    try:
        content_type = audio_file.content_type or 'audio/webm'

        result = stt.recognize(
            audio=audio_file,
            content_type=content_type,
            model='en-US_BroadbandModel'
        ).get_result()

        transcript = ""
        if 'results' in result and len(result['results']) > 0:
            transcript = result['results'][0]['alternatives'][0]['transcript']

        # Auto-analyze sentiment
        sentiment_result = nlu.analyze(
            text=transcript,
            features=Features(sentiment=SentimentOptions())
        ).get_result()
        sentiment = sentiment_result['sentiment']['document']['label'].capitalize()
        score = round(sentiment_result['sentiment']['document']['score'], 2)

        return jsonify({
            'transcript': transcript,
            'sentiment': sentiment,
            'score': score
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
