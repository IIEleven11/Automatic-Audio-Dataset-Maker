import asyncio
from deepgram import Deepgram
import os


DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
if not DEEPGRAM_API_KEY:
    DEEPGRAM_API_KEY = input("Enter your Deepgram API key: ")
    
print("API key:", DEEPGRAM_API_KEY)
async def test_deepgram_connection(api_key):
    try:
        # Initialize the Deepgram client
        dg_client = Deepgram(api_key)

        # Make a simple API call, like getting the list of projects
        response = await dg_client.get_projects()

        if response.status == 200:
            print("Successfully connected to Deepgram API!")
            print("Projects:", response.json)
            return True
        else:
            print(f"Failed to connect. Status code: {response.status}")
            print("Response:", response.json)
            return False

    except Exception as e:
        print(f"An error occurred while connecting to Deepgram: {e}")
        return False

# Your Deepgram API key
DEEPGRAM_API_KEY = "DEEPGRAM_API_KEY"

# Run the test
asyncio.run(test_deepgram_connection(DEEPGRAM_API_KEY))