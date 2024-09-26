from twikit import Client
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

client = Client("en-US")

async def login():
    await client.login(
        auth_info_1=os.getenv("USERNAME"),
        auth_info_2=os.getenv("EMAIL"),
        password=os.getenv("PASSWORD"),    )

async def search_tweet():
    tweets = await client.search_tweet("bitcoin", "Latest")

    return tweets


async def main():
    await login()
    tweets = await search_tweet()
    for tweet in tweets:
        print(
        tweet.user.name,
        tweet.text,
        tweet.created_at
    )


asyncio.run(main())