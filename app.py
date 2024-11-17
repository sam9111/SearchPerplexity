from flask import Flask, request, jsonify
import logging
import time
import os
from collections import defaultdict
# from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path
import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Instead, set the API key directly
# api_key = "your_openaikey_here"

# print(f"API key loaded (last 4 chars): ...{api_key[-4:]}")

# client = OpenAI(api_key=api_key)

PERPLEXITY_TOKEN = os.getenv("PERPLEXITY_TOKEN")

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store for aggregating messages
message_buffer = defaultdict(list)
last_print_time = defaultdict(float)
# AGGREGATION_INTERVAL = 30  # seconds
AGGREGATION_INTERVAL = 10  # seconds

# Add at the top with other global variables
notification_cooldowns = defaultdict(float)
# NOTIFICATION_COOLDOWN = 300  # 5 minutes cooldown between notifications for each session
NOTIFICATION_COOLDOWN = 5

# Add these near the top of the file, after the imports
# if os.getenv('HTTPS_PROXY'):
#     os.environ['OPENAI_PROXY'] = os.getenv('HTTPS_PROXY')

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_drinking_intent(text):
    """Analyze text for drinking intent using OpenAI"""
    # try:
    #     # Add debug logging
    #     logger.info("Attempting to connect to OpenAI API...")
    #     if not api_key:
    #         raise ValueError("OpenAI API key is not set")
        
    #     # Only log the last 4 characters of the API key for security
    #     key_preview = f"...{api_key[-4:]}" if api_key else "None"
    #     logger.info(f"API key check (last 4 chars): {key_preview}")
        
    #     response = client.chat.completions.create(
    #         model="gpt-4",
    #         messages=[
    #             {"role": "system", "content": "You are an AI that analyzes conversations to detect if someone is planning to drink alcohol. Respond with 'YES' if you detect intent to drink alcohol, and 'NO' if you don't."},
    #             {"role": "user", "content": f"Analyze this conversation for intent to drink alcohol: {text}"}
    #         ],
    #         temperature=0.7,
    #         max_tokens=50,
    #         timeout=30  # Add timeout parameter
    #     )
        
    #     answer = response.choices[0].message.content.strip().upper()
    #     logger.info(f"Successfully received response from OpenAI: {answer}")
    #     return answer == "YES"
    # except Exception as e:
    #     logger.error(f"Error analyzing drinking intent: {str(e)}")
    #     logger.error(f"Error type: {type(e).__name__}")
    #     # Print full traceback for debugging
    #     import traceback
    #     logger.error(f"Full traceback: {traceback.format_exc()}")
    #     return False
    return "YES"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_search_intent(text):
    return "search" in text

def print_aggregated_messages(session_id):
    """Print aggregated messages for logging purposes only"""
    if not message_buffer[session_id]:
        return
    
    # Sort messages by start time
    sorted_messages = sorted(message_buffer[session_id], key=lambda x: x['start'])
    
    # Combine all text
    combined_text = ' '.join(msg['text'] for msg in sorted_messages if msg['text'])
    time_range = f"{sorted_messages[0]['start']:.2f}s - {sorted_messages[-1]['end']:.2f}s"
    
    # Just log the transcript without analyzing
    logger.info(f"\n=== Transcript chunk ({time_range}) ===\n{combined_text}\n")
    
    # Clear buffer after processing
    message_buffer[session_id].clear()

def search_perplexity(text):

    print('Inside search_perplexity')

    url = "https://api.perplexity.ai/chat/completions"

    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "You will be given a bunch of text. Extract the relevant question from that and return the answer. Be precise and concise."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "max_tokens": "Optional",
        "temperature": 0.2,
        "top_p": 0.9,
        "return_citations": True,
        "search_domain_filter": ["perplexity.ai"],
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    return response.text
    
def ask_groq(text):

    print('Inside ask_groq')

    client = Groq()
    chat_completion = client.chat.completions.create(
        #
        # Required parameters
        #
        messages=[
            # Set an optional system message. This sets the behavior of the
            # assistant and can be used to provide specific instructions for
            # how it should behave throughout the conversation.
            {
                "role": "system",
                "content": "You are a helpful assistant. You will be given a bunch of text. Extract the relevant question from that and return the answer. Be precise and concise."
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": text,
            }
        ],

        # The language model which will generate the completion.
        model="llama3-8b-8192",

        #
        # Optional parameters
        #

        # Controls randomness: lowering results in less random completions.
        # As the temperature approaches zero, the model will become deterministic
        # and repetitive.
        temperature=0.5,

        # The maximum number of tokens to generate. Requests can use up to
        # 32,768 tokens shared between prompt and completion.
        max_tokens=1024,

        # Controls diversity via nucleus sampling: 0.5 means half of all
        # likelihood-weighted options are considered.
        top_p=1,

        # A stop sequence is a predefined or user-specified text string that
        # signals an AI to stop generating content, ensuring its responses
        # remain focused and concise. Examples include punctuation marks and
        # markers like "[end]".
        stop=None,

        # If set, partial message deltas will be sent.
        stream=False,
    )

    # Print the completion returned by the LLM.
    print(chat_completion.choices[0].message.content)

    return chat_completion.choices[0].message.content

@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == 'POST':
        # Log incoming request
        logger.info("Received webhook POST request")
        data = request.json
        logger.info(f"Received data: {data}")
        
        # Extract session ID and segments
        session_id = data.get('session_id')
        if not session_id:
            logger.error("No session_id provided in request")
            return jsonify({"status": "error", "message": "No session_id provided"}), 400
            
        segments = data.get('segments', [])
        logger.info(f"Processing session_id: {session_id}, number of segments: {len(segments)}")
        
        current_time = time.time()
        
        # Check notification cooldown for this session
        time_since_last_notification = current_time - notification_cooldowns[session_id]
        if time_since_last_notification < NOTIFICATION_COOLDOWN:
            logger.info(f"Notification cooldown active for session {session_id}. {NOTIFICATION_COOLDOWN - time_since_last_notification:.0f}s remaining")
            return jsonify({"status": "success"}), 200
        
        for segment in segments:
            if segment['text']:  # Only store non-empty segments
                message_buffer[session_id].append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'],
                    'speaker': segment['speaker']
                })
                logger.info(f"Added segment text for session {session_id}: {segment['text']}")
        
        # Check if it's time to process messages
        time_since_last = current_time - last_print_time[session_id]
        logger.info(f"Time since last process: {time_since_last}s (threshold: {AGGREGATION_INTERVAL}s)")
        
        if time_since_last >= AGGREGATION_INTERVAL and message_buffer[session_id]:
            logger.info(f"Processing aggregated messages for session {session_id}...")
            sorted_messages = sorted(message_buffer[session_id], key=lambda x: x['start'])
            combined_text = ' '.join(msg['text'] for msg in sorted_messages if msg['text'])
            logger.info(f"Analyzing combined text for session {session_id}: {combined_text}")

            if analyze_search_intent(combined_text.lower()):
                logger.warning(f"Search intent detected for session {session_id}!")
                notification_cooldowns[session_id] = current_time

                # response_text = search_perplexity(combined_text)
                response_text = ask_groq(combined_text)

                return jsonify({
                    "message": f"Groq says: {response_text}"
                }), 200
            
            # Clear the buffer immediately after combining text
            message_buffer[session_id].clear()
            last_print_time[session_id] = current_time
            
        return jsonify({"status": "success"}), 200

@app.route('/webhook/setup-status', methods=['GET'])
def setup_status():
    try:
        # Always return true for setup status
        return jsonify({
            "is_setup_completed": True
        }), 200
    except Exception as e:
        logger.error(f"Error checking setup status: {str(e)}")
        return jsonify({
            "is_setup_completed": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
