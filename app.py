from flask import Flask, render_template, request, jsonify
from ibm_watson import NaturalLanguageUnderstandingV1, AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions

# -------------------------------
# Initialize Flask app
# -------------------------------
app = Flask(__name__)

# -------------------------------
# IBM Watson NLU (Sentiment Analysis)
# -------------------------------
NLU_API_KEY = "eUcO0M_4jov53yc60E50eGEq4VSiY5VW13xMoZLRpRYu"
NLU_API_URL = "https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/81809b0d-2619-4f6f-829d-a4415f14b980"

nlu_auth = IAMAuthenticator(NLU_API_KEY)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=nlu_auth
)
nlu.set_service_url(NLU_API_URL)

# -------------------------------
# IBM Watson Assistant (Chatbot)
# -------------------------------
ASSIST_API_KEY = "qKFbgAauMgoKrGZXtYVk5b_YyqPIfT4wgxTQyxNRgfD2"
ASSIST_URL = "https://api.au-syd.assistant.watson.cloud.ibm.com/instances/50eadfe1-94c4-4df9-8423-eb737ee43456"
ASSISTANT_ID = "1e08c5cb-c1fb-4415-8636-2a45e86bba10"

assist_auth = IAMAuthenticator(ASSIST_API_KEY)
assistant = AssistantV2(
    version='2021-11-27',
    authenticator=assist_auth
)
assistant.set_service_url(ASSIST_URL)

# -------------------------------
# Routes
# -------------------------------

@app.route('/')
def index():
    return render_template('index.html')

# Sentiment analysis route
@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text'].strip()

    if len(text) < 5:
        return render_template(
            'index.html',
            text=text,
            sentiment="⚠️ Please enter a longer sentence for better analysis.",
            score="–"
        )

    try:
        response = nlu.analyze(
            text=text,
            features=Features(sentiment=SentimentOptions())
        ).get_result()

        sentiment = response['sentiment']['document']['label']
        score = response['sentiment']['document']['score']

        return render_template(
            'index.html',
            text=text,
            sentiment=sentiment.capitalize(),
            score=round(score, 2)
        )

    except Exception as e:
        return render_template(
            'index.html',
            text=text,
            sentiment=f"❌ Error: {str(e)}",
            score="–"
        )

# Chatbot route (AJAX endpoint)
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']

    try:
        response = assistant.message_stateless(
            assistant_id=ASSISTANT_ID,
            input={'text': user_input}
        ).get_result()

        # Extract chatbot reply text
        reply = response['output']['generic'][0]['text']
        return jsonify({'reply': reply})

    except Exception as e:
        return jsonify({'reply': f"⚠️ Error connecting to Watson Assistant: {str(e)}"})

# -------------------------------
# Run the app
# -------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
