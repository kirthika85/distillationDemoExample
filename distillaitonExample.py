import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import openai

# Site-specific configurations (keep as is)
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
    """Extract company name and quarter from URL and transcript"""
    # Try to extract from URL
    company_name = ""
    if "marketbeat.com/earnings/reports" in url:
        url_parts = url.split("/")
        if len(url_parts) > 5:
            company_info = url_parts[5].replace("-stock", "").replace("-co", "").replace("-inc", "")
            company_name = company_info.title()
    
    # If we couldn't get it from URL, try first few lines of transcript
    if not company_name and len(transcript) > 200:
        first_chunk = transcript[:200].lower()
        common_names = ["apple", "microsoft", "amazon", "google", "alphabet", "meta", "tesla", 
                       "intel", "amd", "nvidia", "coca-cola", "pepsi", "walmart", "target", 
                       "johnson & johnson", "pfizer", "merck"]
        for name in common_names:
            if name in first_chunk:
                company_name = name.title()
                break
    
    return company_name

def analyze_overall_sentiment(transcript, api_key, company_name=""):
    """Analyze the overall sentiment of the transcript using the LLM directly"""
    client = openai.OpenAI(api_key=api_key)
    
    # If transcript is too long, process it in chunks of 12000 characters
    max_chunk_size = 12000
    sentiment_results = []
    
    # Create a more specific prompt for financial analysis
    finance_prompt = f"""You are a financial analyst specializing in earnings call assessment. 
Analyze the following {company_name} earnings call transcript and determine the OVERALL SENTIMENT.

When analyzing, consider these key factors:
1. Forward guidance and management outlook
2. Revenue, profit, and margin trends
3. Market share and competitive position
4. Analyst questions and management responses
5. Any mentions of stock price, market reaction, or valuation
6. Language around restructuring, layoffs, or cost-cutting
7. References to macroeconomic challenges or tailwinds

Provide your analysis in this EXACT format:
1. SENTIMENT: [Positive/Negative/Mixed/Cautiously Positive/Cautiously Negative]
2. KEY FACTORS: Brief bullet points of the most influential aspects
3. CONFIDENCE: [High/Medium/Low]

Earnings calls are considered NEGATIVE when they include:
- Missed targets or lowered guidance
- Declining metrics or negative growth
- Defensive management responses
- Concerns about competition or market challenges
- Cost-cutting as a primary focus
- Restructuring or layoffs

Earnings calls are considered POSITIVE when they include:
- Exceeded expectations and raised guidance
- Strong growth metrics and expanding margins
- Confident and forward-looking management comments
- Market share gains and competitive advantages
- Innovation and new product success

TRANSCRIPT EXCERPT:
"""
    
    if len(transcript) <= max_chunk_size:
        # Single chunk analysis
        full_prompt = finance_prompt + transcript
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.2,
                max_tokens=500
            )
            analysis = response.choices[0].message.content
            sentiment_results.append(analysis)
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
            return None
    else:
        # Multi-chunk analysis
        chunks = []
        # First chunk includes the beginning (often contains important overview)
        chunks.append(transcript[:max_chunk_size])
        
        # Add one or more middle chunks if very long
        if len(transcript) > max_chunk_size * 2:
            middle_start = len(transcript) // 2 - max_chunk_size // 2
            chunks.append(transcript[middle_start:middle_start + max_chunk_size])
        
        # Last chunk includes the end (often contains guidance and Q&A)
        chunks.append(transcript[-max_chunk_size:])
        
        for i, chunk in enumerate(chunks):
            chunk_prompt = finance_prompt + f"\n[CHUNK {i+1} of {len(chunks)}]: " + chunk
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": chunk_prompt}],
                    temperature=0.2,
                    max_tokens=500
                )
                analysis = response.choices[0].message.content
                sentiment_results.append(analysis)
            except Exception as e:
                st.error(f"OpenAI API error on chunk {i+1}: {e}")
                continue
    
    # If we processed multiple chunks, ask LLM to consolidate the results
    if len(sentiment_results) > 1:
        consolidation_prompt = f"""As a financial analyst, review these separate analyses of the same {company_name} earnings call and provide a FINAL OVERALL ASSESSMENT.

Analysis 1: {sentiment_results[0]}

Analysis 2: {sentiment_results[1]}

{f"Analysis 3: {sentiment_results[2]}" if len(sentiment_results) > 2 else ""}

Based on these analyses, provide your final assessment in this EXACT format:
1. SENTIMENT: [Positive/Negative/Mixed/Cautiously Positive/Cautiously Negative]
2. KEY FACTORS: Brief bullet points of the most influential aspects
3. CONFIDENCE: [High/Medium/Low]
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": consolidation_prompt}],
                temperature=0.2,
                max_tokens=500
            )
            final_analysis = response.choices[0].message.content
            return final_analysis
        except Exception as e:
            st.error(f"OpenAI API error during consolidation: {e}")
            # Return the first analysis if consolidation fails
            return sentiment_results[0]
    
    # If we only had one chunk, return its analysis
    return sentiment_results[0] if sentiment_results else None

# Streamlit UI
st.title("Earnings Call Sentiment Analyzer")

url = st.text_input("Enter transcript URL:")
                  
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

    # Extract company name for more specific analysis
    company_name = extract_company_info(url, transcript)
    
    with st.spinner("Analyzing overall sentiment with GPT-4..."):
        analysis_result = analyze_overall_sentiment(transcript, api_key, company_name)
    
    if analysis_result:
        st.subheader("Sentiment Analysis Results")
        
        # Parse and display the structured result
        lines = analysis_result.split('\n')
        sentiment_line = next((line for line in lines if "SENTIMENT:" in line), "")
        
        if sentiment_line:
            # Extract just the sentiment classification
            sentiment = sentiment_line.split("SENTIMENT:")[1].strip()
            if any(term in sentiment.lower() for term in ["negative", "cautiously negative"]):
                sentiment_color = "red"
            elif any(term in sentiment.lower() for term in ["positive", "cautiously positive"]):
                sentiment_color = "green"
            else:
                sentiment_color = "orange"  # Mixed or other
            
            st.markdown(f"### Overall Sentiment: <span style='color:{sentiment_color}'>{sentiment}</span>", unsafe_allow_html=True)
        
        # Display the full analysis with formatting
        st.markdown("### Detailed Analysis")
        st.markdown(analysis_result.replace("1. ", "**").replace("2. ", "**").replace("3. ", "**").replace(":", ":**"))
        
        # Add a visualization if you want
        if "SENTIMENT:" in analysis_result:
            st.markdown("### Key Insights")
            st.info("This analysis was performed by examining the entire transcript in context, considering financial indicators, guidance, and analyst interactions.")
    else:
        st.error("No sentiment results returned")
