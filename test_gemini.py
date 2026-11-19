
from google import genai
from config import GEMINI_API_KEY


def test_gemini():

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing. Check your .env file."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    try:
        print("Connecting to Gemini...")

        response = client.interactions.create(
            model="gemini-3.8-flash",
            input="Reply with: Gemini connection successful!"
        )

        print("\nGemini Response:")
        print(response.output_text)

        print("\nGemini API connection successful!")

    except Exception as error:
        print(f"\nGemini API Error: {error}")

    finally:
        client.close()


if __name__ == "__main__":
    test_gemini()
