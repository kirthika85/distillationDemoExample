import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
import openai

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
    

def scrape_transcript(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Seeking Alpha's transcript structure
        transcript_container = soup.find('div', class_='article-section')
        if not transcript_container:
            return ""
            
        # Extract all paragraphs with transcript content
        paragraphs = transcript_container.find_all('p', class_='paragraph')
        transcript = ' '.join([p.get_text(strip=True) for p in paragraphs])
        
        # Remove irrelevant sections
        transcript = transcript.replace("Operator", "").replace("Q&A Session", "")
        return transcript.strip()
    
    except Exception as e:
        st.error(f"Scraping error: {str(e)}")
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

url = st.text_input("Enter the transcript URL:", 
                   value="https://seekingalpha.com/article/4776029-snap-on-incorporated-sna-q1-2025-earnings-call-transcript")

if st.button("Analyze Sentiment"):
    if not url:
        st.warning("Please enter a transcript URL.")
        st.stop()

    api_key = st.secrets["OPENAI_API_KEY"]

    transcript = scrape_transcript(url)
    if not transcript:
        st.error("Unable to extract transcript. Seeking Alpha structure may have changed.")
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
