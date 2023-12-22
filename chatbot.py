from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
from chatterbot.logic import BestMatch, MathematicalEvaluation, TimeLogicAdapter
from chatterbot.comparisons import JaccardSimilarity
from chatterbot.conversation import Statement
from textblob import TextBlob  # For sentiment analysis (requires: pip install textblob)
import requests  # For API integration

# Create a new instance of a ChatBot with specified logic adapters
chatbot = ChatBot(
    'MyChatBot',
    logic_adapters=[
        'chatterbot.logic.BestMatch',
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter'
    ],
    statement_comparison_function=JaccardSimilarity,
    preprocessors=[
        'chatterbot.preprocessors.clean_whitespace'
    ],
    storage_adapter='chatterbot.storage.SQLStorageAdapter'
)

# Create a new trainer for the chatbot
trainer = ChatterBotCorpusTrainer(chatbot)

# Train the chatbot based on the English corpus
trainer.train('chatterbot.corpus.english')

# Train the bot with custom data
trainer_list = ListTrainer(chatbot)
trainer_list.train([
    'How can I help you?',
    'I want to create a chatbot',
    'Have you read the documentation?'
])

# API integration example (replace with your own API)
def fetch_data_from_api():
    response = requests.get('https://api.example.com/data')
    if response.status_code == 200:
        return response.json()
    return None

# Run the chatbot in a loop
while True:
    try:
        user_input = input('You: ')
        response = chatbot.get_response(user_input)

        # Sentiment analysis
        blob = TextBlob(user_input)
        sentiment_score = blob.sentiment.polarity

        # Handle ambiguous queries
        if sentiment_score < -0.5:
            print("Bot: It seems like you're expressing dissatisfaction. Can you provide more details?")
        elif sentiment_score > 0.5:
            print("Bot: That's great! How can I assist you further?")
        else:
            print('Bot:', response)

        # Context-awareness (you can implement a more sophisticated context management system here)

        # Example API integration usage
        data = fetch_data_from_api()
        if data:
            print('Bot: Here is some data from the API:', data)
        else:
            print('Bot: Sorry, I could not retrieve data from the API.')

    except (KeyboardInterrupt, EOFError, SystemExit):
        break
