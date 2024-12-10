import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Step 1: Define Chatbot Scope and Dataset
data = [
    # Intent: Greetings
    {"intent": "greeting", "patterns": ["Hi", "Hello", "Hey", "Good morning", "Good afternoon"], "responses": ["Hello!", "Hi there!", "Good morning!", "Hey! How can I assist you?"]},
    
    # Intent: Goodbye
    {"intent": "goodbye", "patterns": ["Bye", "See you", "Goodbye", "Take care", "Catch you later"], "responses": ["Goodbye!", "See you later!", "Take care!", "Catch you soon!"]},
    
    # Intent: Thanks
    {"intent": "thanks", "patterns": ["Thanks", "Thank you", "Thanks a lot", "I appreciate it"], "responses": ["You're welcome!", "No problem!", "Glad to help!", "Anytime!"]},
    
    # Intent: Unknown
    {"intent": "unknown", "patterns": ["What's up?", "How are you?", "What can you do?", "Tell me something interesting"], "responses": ["I'm just a chatbot, but I'm here to help!", "I'm doing great, thanks for asking!", "I can help with many things, try asking me!"]},
    
    # Intent: Small Talk
    {"intent": "small_talk", "patterns": ["How are you?", "What’s new?", "How’s life?", "What’s going on?", "What are you up to?"], "responses": ["I'm just a bot, but I'm doing fine!", "Life's good, how about you?", "I'm functioning perfectly, thanks!"]},
    
    # Intent: About Bot
    {"intent": "about_bot", "patterns": ["What is your name?", "Who are you?", "What do you do?", "Tell me about yourself", "What’s your purpose?"], "responses": ["I am Chatbot, here to assist you!", "I am an AI chatbot created to help you with various tasks.", "I am a virtual assistant, ask me anything!"]},
    
    # Intent: Weather queries (Added more diverse patterns)
    {"intent": "weather", "patterns": ["What's the weather like?", "Tell me the weather", "Is it raining?", "How’s the weather today?", "What's the temperature?", "Will it rain today?", "How's the weather looking?"], "responses": ["Sorry, I can't check the weather right now.", "You can check your local weather app for that.", "I don’t have weather information, but I can tell you a joke!"]},
    
    # Intent: Jokes (More humorous patterns)
    {"intent": "jokes", "patterns": ["Tell me a joke", "Make me laugh", "Do you know any jokes?", "Give me a funny one", "Tell me something funny", "I need a good laugh"], "responses": ["Why don't skeletons fight each other? They don’t have the guts!", "I told my computer I needed a break, now it won't stop sending me Kit-Kats.", "Why don’t programmers like nature? It has too many bugs!", "What do you call fake spaghetti? An impasta!"]},
    
    # Intent: User Experience
    {"intent": "user_experience", "patterns": ["How am I doing?", "How's my progress?", "What can I improve on?", "Is everything okay?"], "responses": ["You're doing great!", "Keep it up!", "You’re making good progress, keep going!"]},
]

# Step 2: Data Preparation
X = []
y = []
for item in data:
    X.extend(item["patterns"])
    y.extend([item["intent"]] * len(item["patterns"]))

# Train-test split
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Train Intent Classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Step 3: Load Hugging Face's DialoGPT for Dynamic Responses
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Helper function to predict intent
def predict_intent(user_input):
    user_vect = vectorizer.transform([user_input])
    intent = classifier.predict(user_vect)[0]
    print(f"Predicted intent: {intent}")  # Debugging statement
    return intent

# Generate response
def generate_response(user_input, intent):
    for item in data:
        if item["intent"] == intent:
            return random.choice(item["responses"])
    # Fallback to DialoGPT for unrecognized intents
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

# Step 4: Chat Loop
print("Chatbot: Hi! Type 'bye' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "bye":
        print("Chatbot: Goodbye!")
        break

    # Predict intent and generate response
    intent = predict_intent(user_input)
    response = generate_response(user_input, intent)
    print(f"Chatbot: {response}")
