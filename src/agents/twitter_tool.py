## twitter_tool.py
import nest_asyncio
nest_asyncio.apply()

import asyncio
from langchain_core.tools import tool
from twikit import Client
from arcade_x import PostTweet, DeleteTweetById, SearchRecentTweetsByUsername, SearchRecentTweetsByKeywords, LookupSingleUserByUsername

# Hardcoded credentials (replace with your actual credentials)
USERNAME = "dylansz_"
EMAIL = "dylanszejnblum@gmail.com"
PASSWORD = "@testo123"

# Initialize Twikit client
client = Client('en-US')

# Login to Twitter
async def login():
    await client.login(
        auth_info_1=USERNAME,
        auth_info_2=EMAIL,
        password=PASSWORD
    )

asyncio.run(login())

@tool
def post_tweet(tweet_text: str) -> str:
    """Posts a tweet to X (Twitter).
    
    Args:
        tweet_text (str): The text content of the tweet
    
    Returns:
        str: Response message indicating success or failure
    """
    try:
        post_tool = PostTweet()
        result = post_tool.invoke({"tweet_text": tweet_text})
        return f"Tweet posted successfully: {result}"
    except Exception as e:
        return f"Failed to post tweet: {str(e)}"

@tool
def search_user_tweets(username: str, max_results: int = 10) -> str:
    """Searches recent tweets from a specific user.
    
    Args:
        username (str): The username to search tweets from
        max_results (int, optional): Maximum number of tweets to return. Defaults to 10.
    
    Returns:
        str: JSON string containing the found tweets
    """
    try:
        search_tool = SearchRecentTweetsByUsername()
        result = search_tool.invoke({
            "username": username,
            "max_results": max_results
        })
        return result
    except Exception as e:
        return f"Failed to search tweets: {str(e)}"

@tool
def search_tweets_by_keywords(keywords: list[str], phrases: list[str] = None, max_results: int = 10) -> str:
    """Searches recent tweets containing specific keywords or phrases.
    
    Args:
        keywords (list[str]): List of keywords to search for
        phrases (list[str], optional): List of exact phrases to search for
        max_results (int, optional): Maximum number of tweets to return. Defaults to 10.
    
    Returns:
        str: JSON string containing the found tweets
    """
    try:
        search_tool = SearchRecentTweetsByKeywords()
        result = search_tool.invoke({
            "keywords": keywords,
            "phrases": phrases,
            "max_results": max_results
        })
        return result
    except Exception as e:
        return f"Failed to search tweets: {str(e)}"