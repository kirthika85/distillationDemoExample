import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
import openai
import re

# Configure NLTK
nltk.download('punkt')
nltk.download('punkt_tab')

# Site-specific configurations
SITE_CONFIG = {
    'seekingalpha': {
        'container': {'name': 'div', 'class_': 'article-body'},
        'paragraphs': {'name': 'p', 'class_': re.compile(r'^paragraph')},
        'filters': [
            'Sign up here', 'Read more', 'This is a transcript', 
            'Company Participants', 'Operator', 'Q&A Session'
        ]
    },
    'investing.com': {
        'container': {'name': 'div', 'class_': 'articlePage'},
        'paragraphs': {'name': 'p'},
        'filters': ['This article', 'For more information', 'Disclaimer:']
    },
    'marketbeat.com': {
        'container': {'name': 'div', 'class_': 'transcript'},
        'paragraphs': {'name': 'p'},
        'filters': ['Disclaimer', 'Forward-Looking Statements']
    },
    'default': {
        'container': {'name': 'body'},
        'paragraphs': {'name': 'p'},
        'filters': ['Operator', 'Q&A', 'Forward-Looking Statements']
    }
}

def get_site_config(url):
    """Identify which site config to use based on URL"""
    for site in SITE_CONFIG:
        if site in url:
            return SITE_CONFIG[site]
    return SITE_CONFIG['default']

def scrape_transcript(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "DNT": "1"
    }
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        config = get_site_config(url)
        container = soup.find(**config['container']) or soup
        
        paragraphs = container.find_all(**config['paragraphs'])
        transcript = []
        
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and not any(text.startswith(f) for f in config['filters']):
                transcript.append(text)
                
        full_text = ' '.join(transcript)
        return full_text.strip() if len(full_text) > 500 else ""
    
    except Exception as e:
        st.error(f"Scraping error: {str(e)}")
        return ""

def analyze_sentiment_gpt4(text, api_key, chunk_size=15):
    client = openai.OpenAI(api_key=api_key)
    sentences = nltk.sent_tokenize(text)
    all_labels = []
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i+chunk_size]
        prompt = (
            "Classify each sentence's sentiment as ONLY positive, negative, or neutral. "
            "Return EXACTLY a comma-separated list of labels in order. Example: positive, neutral, negative\n\n"
        )
        for j, sentence in enumerate(chunk, 1):
            prompt += f"{j}. {sentence}\n"
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )
            labels = response.choices[0].message.content.split(', ')
            all_labels.extend(labels)
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
            return []
    return all_labels

def get_overall_sentiment(sentiment_counts):
    """Determine overall sentiment based on sentence-level analysis"""
    pos = sentiment_counts['positive']
    neg = sentiment_counts['negative']
    neu = sentiment_counts['neutral']
    total = pos + neg + neu
    
    if total == 0:
        return "Unknown"
    
    pos_ratio = pos / total
    neg_ratio = neg / total
    
    # Adjusted thresholds for negative classification (from 0.4 to 0.35)
    if pos_ratio >= 0.4:  # Strong positive majority
        if neg_ratio >= 0.25:  # More sensitive to negative presence
            return "Mixed / Cautiously Positive"
        return "Positive"
    elif neg_ratio >= 0.35:  # Lowered threshold for negative classification (was 0.4)
        if pos_ratio >= 0.25:
            return "Mixed / Cautiously Negative"
        return "Negative"  # Now catches cases like Intel
    else:  # No clear majority
        if pos > neg:
            return "Mixed / Leaning Positive"
        elif neg > pos:
            return "Mixed / Leaning Negative"
        return "Neutral / Balanced"


# Streamlit UI
st.title("Multi-Site Earnings Call Sentiment Analyzer")

url = st.text_input("Enter transcript URL:", 
                   value="https://www.investing.com/news/transcripts/earnings-call-transcript-badger-meter-q1-2025-beats-earnings-expectations-93CH-3991443")

if st.button("Analyze Sentiment"):
    if not url:
        st.warning("Please enter a transcript URL.")
        st.stop()

    api_key = st.secrets.get("OPENAI_API_KEY", "your-api-key-here")

    with st.spinner("Scraping transcript..."):
        transcript = scrape_transcript(url)
    
    if not transcript:
        st.error("Failed to extract transcript. Site structure may have changed.")
        st.stop()

    st.subheader("Transcript Preview")
    st.write(transcript[:500] + "...")

    with st.spinner("Analyzing sentiment with GPT-4..."):
        labels = analyze_sentiment_gpt4(transcript, api_key)
    
    if labels:
        sentiment_counts = {
            "positive": labels.count('positive'),
            "neutral": labels.count('neutral'),
            "negative": labels.count('negative')
        }
        
        # Calculate overall sentiment
        overall_sentiment = get_overall_sentiment(sentiment_counts)
        
        # Display results
        st.subheader("Sentiment Analysis Results")
        
        # Overall sentiment
        st.markdown(f"**Overall Sentiment:** {overall_sentiment}")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Positive Sentences", sentiment_counts['positive'])
        col2.metric("Neutral Sentences", sentiment_counts['neutral'])
        col3.metric("Negative Sentences", sentiment_counts['negative'])
        
        # Sample analysis
        st.subheader("Sample Sentence Analysis")
        sentences = nltk.sent_tokenize(transcript)
        sample_size = min(5, len(sentences))
        for i in range(sample_size):
            st.markdown(f"""
            **Sentence {i+1}:**  
            {sentences[i]}  
            **Sentiment:** {labels[i].capitalize()}
            """)
            st.divider()
    else:
        st.error("No sentiment results returned")
