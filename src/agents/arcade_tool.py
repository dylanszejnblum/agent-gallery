from langchain_core.tools import tool
from arcadepy import Arcade
import os
import time
from typing import Optional, Dict, Any
import backoff
from datetime import datetime, timezone

# Configuration
USER_ID = os.environ.get("ARCADE_USER_ID", "dylan@polh.io")  # Fallback for development
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def get_arcade_client() -> Arcade:
    """Creates and returns an Arcade client"""
    return Arcade()

async def execute_with_auth(
    client: Arcade,
    tool_name: str,
    inputs: Dict[str, Any],
    user_id: str = USER_ID
) -> Any:
    """Execute an Arcade tool with authentication handling
    
    Args:
        client: Arcade client instance
        tool_name: Name of the Arcade tool to execute
        inputs: Dictionary of input parameters for the tool
        user_id: User ID for authentication
        
    Returns:
        Tool execution results
    """
    # Handle authorization
    auth_response = client.tools.authorize(
        tool_name=tool_name,
        user_id=user_id,
    )

    if auth_response.status != "completed":
        print(f"Authorization required. Please visit: {auth_response.authorization_url}")
        client.auth.wait_for_completion(auth_response)

    # Execute the tool with retries
    @backoff.on_exception(backoff.expo, Exception, max_tries=MAX_RETRIES)
    def execute():
        return client.tools.execute(
            tool_name=tool_name,
            inputs=inputs,
            user_id=user_id,
        )

    return execute()

async def get_all_tweets(
    client: Arcade,
    username: str,
    max_results: int = 100,
    user_id: str = USER_ID
) -> list:
    """Fetch all available tweets for a given username using pagination
    
    Args:
        client: Arcade client instance
        username: Twitter username to fetch tweets for
        max_results: Maximum number of results per page
        user_id: User ID for authentication
        
    Returns:
        list: All collected tweets
    """
    all_tweets = []
    next_token = None
    
    while True:
        inputs = {"username": username, "max_results": max_results}
        if next_token:
            inputs["next_token"] = next_token
            
        response = await execute_with_auth(
            client,
            "X.SearchRecentTweetsByUsername",
            inputs,
            user_id
        )
        
        tweets_data = response.output.value.get('data', [])
        all_tweets.extend(tweets_data)
        
        next_token = response.output.value.get("meta", {}).get("next_token")
        if not next_token:
            break
            
    return all_tweets

@tool
async def post_tweet(tweet_text: str, user_id: str = USER_ID) -> str:
    """Posts a tweet to X (Twitter).
    
    Args:
        tweet_text (str): The text content of the tweet
        user_id (str): The user's email for authorization
    
    Returns:
        str: Response message indicating success or failure
    """
    try:
        client = get_arcade_client()
        response = await execute_with_auth(
            client,
            "Twitter.PostTweet",
            {"tweet_text": tweet_text},
            user_id
        )
        return f"Tweet posted successfully: {response.output.value}"
    except Exception as e:
        return f"Failed to post tweet: {str(e)}"

@tool
async def fetch_user_tweets(
    username: str,
    user_id: str = USER_ID,
    max_results: int = 10
) -> str:
    """Fetches recent tweets from a specific user.
    
    Args:
        username (str): The username to fetch tweets from
        user_id (str): The user's email for authorization
        max_results (int, optional): Maximum number of tweets to return. Defaults to 10.
    
    Returns:
        str: JSON string containing the found tweets
    """
    try:
        client = get_arcade_client()
        tweets = await get_all_tweets(
            client,
            username,
            max_results=max_results,
            user_id=user_id
        )
        
        # Format tweets for return
        formatted_tweets = []
        for tweet in tweets:
            formatted_tweets.append({
                'id': tweet.get('id'),
                'text': tweet.get('text'),
                'created_at': tweet.get('created_at'),
                'url': tweet.get('tweet_url')
            })
            
        return formatted_tweets
    except Exception as e:
        return f"Failed to fetch tweets: {str(e)}" 