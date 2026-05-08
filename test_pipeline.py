import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Must set env vars before import
import sys
sys.path.insert(0, ".")
from new_helper import process_transcript, convert_non_null_values_to_text

async def main():
    transcript = "Hi I am Steve Black, my bathroom sink is not working."
    print(f"Testing pipeline with: '{transcript}'\n")
    result = await process_transcript(transcript)
    result = await convert_non_null_values_to_text(result)
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
