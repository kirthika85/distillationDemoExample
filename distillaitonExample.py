import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import openai
import json

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
    """Scrape transcript from the URL"""
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

def extract_company_info(url, transcript):
    """Improved company name extraction"""
    # From URL
    if "marketbeat.com/earnings/reports" in url:
        parts = url.split("/")
        if len(parts) > 5:
            return parts[5].replace("-stock", "").replace("-co", "").replace("-inc", "").title()
    
    # From transcript content
    first_200 = transcript[:200].lower()
    patterns = {
        "Intel": r"\bintel\b",
        "Microsoft": r"\bmicrosoft\b",
        "Apple": r"\bapple\b",
        "Tesla": r"\btesla\b"
    }
    
    for name, pattern in patterns.items():
        if re.search(pattern, first_200):
            return name
    
    return "Company"

def analyze_overall_sentiment(transcript, api_key, company_name=""):
    client = openai.OpenAI(api_key=api_key)
    critical_negatives = [
        "operating loss", "guidance cut", "dividend reduction", "margin compression",
        "stock decline", "subscriber decline", "churn rate", "supply chain constraints",
        "tariff impacts", "competitive threats", "inventory glut", "restructuring charges",
        "regulatory hurdles", "pricing pressure", "order cancellations", "bottom of cycle"
    ]
    prompt = f"""Analyze {company_name} earnings call transcript in JSON Format:
1. **Negative Classification** (REQUIRE 2+):
   - EPS/revenue miss + guidance cut
   - Stock decline >5% post-earnings
   - Operating margin decline + negative FCF
   - Critical risks: {', '.join(critical_negatives)}

2. **Mixed Classification** (REQUIRE both):
   - At least one strong positive (e.g., revenue beat, record cash flow, dividend increase, strong project pipeline)
   - At least one strong negative (e.g., EPS miss, margin contraction, bottom-of-cycle margins, guidance uncertainty, missed analyst expectations, significant headwinds, or negative YoY growth)
   - If both are present, classify as Mixed, even if positives are emphasized in the call.

3. **Positive Classification**:
   - All major metrics beat, guidance raised, no significant negatives or headwinds discussed.

4. **Response Format JSON**:
{{
  "sentiment": "Positive/Mixed/Negative",
  "confidence": 0-1,
  "key_factors": ["Specific metric 1", "Metric 2", ...],
  "negative_triggers": []
}}

Transcript: {transcript[:12000]}"""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        analysis = json.loads(response.choices[0].message.content)

        # Post-processing for Mixed
        positive_terms = ["record", "growth", "beat", "increase", "raised guidance", "cash flow", "dividend", "project pipeline"]
        negative_terms = ["miss", "margin compression", "bottom of cycle", "guidance cut", "decline", "headwind", "uncertainty", "missed expectations", "negative growth"]

        pos_count = sum(1 for t in positive_terms if t in transcript.lower())
        neg_count = sum(1 for t in negative_terms if t in transcript.lower())

        if pos_count >= 2 and neg_count >= 2:
            analysis["sentiment"] = "Mixed"
            analysis["confidence"] = max(analysis.get("confidence", 0.7), 0.7)
            if "Balanced positive/negative indicators" not in analysis["key_factors"]:
                analysis["key_factors"].insert(0, "Balanced positive/negative indicators")
        return analysis

    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None

# Streamlit UI
st.title("Earnings Call Analyzer")

url = st.text_input("Enter Transcript URL:", 
                   placeholder="https://www.marketbeat.com/earnings/reports/...")

if st.button("Analyze"):
    if not url:
        st.warning("Please enter a valid URL")
        st.stop()
        
    api_key = st.secrets.get("OPENAI_API_KEY")
    
    with st.spinner("Scraping transcript..."):
        transcript = scrape_transcript(url)
        if not transcript:
            st.error("Transcript extraction failed")
            st.stop()
            
    company_name = extract_company_info(url, transcript)
    
    with st.spinner("Analyzing financial sentiment..."):
        result = analyze_overall_sentiment(transcript, api_key, company_name)
        
    if result:
        color_map = {
            "Negative": "red",
            "Mixed": "orange",
            "Positive": "green"
        }
        sentiment = result.get("sentiment", "Unavailable")
        sentiment_color = color_map.get(sentiment, "gray")
        
        st.subheader("Analysis Results")
        st.markdown(f"""
        <div style='border-left: 5px solid {color_map[result["sentiment"]]}; padding: 1rem;'>
            <h3 style='color:{color_map[result["sentiment"]]};'>{result["sentiment"]}</h3>
            <p>Confidence: {result['confidence']*100:.0f}%</p>
            <p><b>Key Factors:</b></p>
            <ul>{"".join([f"<li>{f}</li>" for f in result['key_factors']])}</ul>
            {f"<p><b>Negative Triggers:</b> {', '.join(result['negative_triggers'])}</p>" if result['negative_triggers'] else ""}
        </div>
        """, unsafe_allow_html=True)
        
