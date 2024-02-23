import speech_recognition as sr
from transformers import pipeline

def record_audio():
    """Record audio from the microphone and return the transcribed text."""
    # Create a recognizer instance
    r = sr.Recognizer()
    
    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Say something!")
        # Listen for audio input
        audio = r.listen(source)

    try:
        # Recognize speech using Google Speech Recognition
        text = r.recognize_google(audio)
        print(text)
        return text
    except sr.UnknownValueError:
        # Speech is unintelligible
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        # Couldn't request results due to network issue or service error
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return None


# Load the pipeline for text generation using GPT-2 model
text_generation_pipeline = pipeline("text-generation", model="gpt2")

# Load the pipeline for sentiment analysis using BERT model
sentiment_analysis_pipeline = pipeline("sentiment-analysis")

def provide_feedback(text):
    """Analyzes the text and provides feedback on grammar, sentiment, and style."""
    # Perform sentiment analysis using BERT model
    sentiment = sentiment_analysis_pipeline(text)[0]['label']
    
    # Generate suggestions for improving the text using GPT-2 model
    generated_text = text_generation_pipeline(text,max_length=100, num_return_sequences=1)[0]['generated_text']
    
    feedback = "**Feedback:**\n"
    feedback += "- Sentiment: {}\n".format(sentiment)
    feedback += "- Style Suggestions: {}\n".format(generated_text)
    
    return feedback

if __name__ == "__main__":
    text = record_audio()
    if text:
        feedback = provide_feedback(text)
        print(feedback)
