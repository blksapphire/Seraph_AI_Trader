# modules/fundamental_analyzer.py
import requests
from transformers import pipeline
import logging

class FundamentalAnalyzer:
    def __init__(self, config):
        self.config = config
        self.api_key = config['fundamental_parameters']['news_api_key']
        model_name = config['fundamental_parameters']['news_sentiment_model']
        # This will download the model on first run
        self.sentiment_pipeline = pipeline('sentiment-analysis', model=model_name)
        logging.info(f"Sentiment analysis model '{model_name}' loaded.")

    def get_news_sentiment(self):
        """Fetches news and calculates a weighted sentiment score."""
        total_score = 0
        article_count = 0
        headlines = []
        
        try:
            for currency in self.config['fundamental_parameters']['currencies_to_monitor']:
                url = f"https://newsapi.org/v2/everything?q={currency}&apiKey={self.api_key}&language=en&sortBy=publishedAt&pageSize=10"
                response = requests.get(url)
                articles = response.json().get('articles', [])
                
                for article in articles:
                    sentiment = self.sentiment_pipeline(article['title'])[0]
                    headlines.append(f"({sentiment['label']}) {article['title']}")
                    if sentiment['label'] == 'positive':
                        total_score += sentiment['score']
                    elif sentiment['label'] == 'negative':
                        total_score -= sentiment['score']
                    article_count += 1
            
            if article_count == 0:
                return {'score': 0, 'narrative': 'No news found'}

            # Average score, normalized to be between -1 and 1
            final_score = total_score / article_count
            return {'score': final_score, 'narrative': headlines[0] if headlines else "Analysis complete"}

        except Exception as e:
            logging.error(f"Failed to get news sentiment: {e}")
            return {'score': 0, 'narrative': 'News API Error'}