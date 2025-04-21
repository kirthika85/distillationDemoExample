import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
import openai

nltk.download('punkt', quiet=True)

def scrape_transcript(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        transcript_containers = soup.find_all(['div', 'section'], class_=['transcript-text', 'article-text', 'transcript-container'])
        if not transcript_containers:
            transcript_containers = soup.find_all('p')
        transcript = ' '.join([container.get_text(strip=True) for container in transcript_containers])
        irrelevant_phrases = ['Video Player', 'Loading Video', 'Transcript', 'Q&A Session']
        for phrase in irrelevant_phrases:
            transcript = transcript.replace(phrase, '')
        return transcript.strip()
    except Exception:
        return ""

def analyze_sentiment_gpt4(text, api_key):
    client = openai.OpenAI(api_key=api_key)
    sentences = nltk.sent_tokenize(text)
    prompt = (
        "Classify each sentence's sentiment as ONLY positive, negative, or neutral. "
        "Return EXACTLY a comma-separated list of labels in order. Example: positive, neutral, negative\n\n"
    )
    for i, sentence in enumerate(sentences, 1):
        prompt += f"{i}. {sentence}\n"
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content.split(', ')
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return []

# --- Streamlit UI ---
st.title("Earnings Call Sentiment Analysis (GPT-4)")

url = st.text_input("Enter the transcript URL:")

if st.button("Analyze Sentiment"):
    if not url:
        st.warning("Please enter a transcript URL.")
        st.stop()

    # Get OpenAI API key from Streamlit secrets
    api_key = st.secrets["OPENAI_API_KEY"]

    transcript = scrape_transcript(url)
    if not transcript:
        st.error("Unable to extract transcript from the provided URL. Please check the URL and try again.")
    else:
        st.subheader("Transcript Sample")
        st.write(transcript[:500] + "...")

        labels = analyze_sentiment_gpt4(transcript, api_key)
        if labels:
            sentiment_counts = {
                "positive": labels.count('positive'),
                "neutral": labels.count('neutral'),
                "negative": labels.count('negative')
            }
            st.subheader("Sentiment Results")
            st.write(f"Positive: {sentiment_counts['positive']}")
            st.write(f"Neutral: {sentiment_counts['neutral']}")
            st.write(f"Negative: {sentiment_counts['negative']}")

            sentences = nltk.sent_tokenize(transcript)
            sample_size = min(5, len(sentences))
            st.subheader("Sample Analysis")
            for i in range(sample_size):
                st.write(f"**Sentence {i+1}:** {sentences[i]}")
                st.write(f"**Sentiment:** {labels[i].capitalize()}\n")
        else:
            st.error("No sentiment results returned.")
