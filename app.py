'''import requests as r

# add review
text = "Everything which is against nature and unnatural is legalized in US in the name of so called choice. Freedom and love.. LGBTQ is its perfect example.. LGBTQ is not a choice or love its a mental sickness which needs proper treatment "

keys = {"text": text}

prediction = r.get("http://127.0.0.1:8000/predict/", params=keys)

results = prediction.json()
print(results["prediction"])
print(results["Probability"])'''

from string import punctuation
# text preprocessing modules
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  
from os.path import dirname, join, realpath
import uvicorn
from fastapi import FastAPI 


app = FastAPI(
    title="Hope Speech Detection",
    description="A simple API that use NLP model to predict the hopefullness in comments towards the LQBTQ Community",
    version="0.1",
)

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


import pickle
pickle_in = open('model.pkl', 'rb')
model = pickle.load(pickle_in)

def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])
    # Optionally, remove stop words
    if remove_stop_words:
        # load stopwords
        stop_words = stopwords.words("english")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    # Return a list of words
    return text


@app.get("/predict")
def predict(text: str):
    """
    A simple function that receive a text content and predict the hopefullness of the content.
    :param review:
    :return: prediction, probabilities
    """
    # clean the review
    cleanedText = text_cleaning(text)
    
    # perform prediction
    prediction = model.predict([cleanedText])
    output = int(prediction[0])
    probas = model.predict_proba([cleanedText])
    output_probability = "{:.2f}".format(float(probas[:, output]))
    
    # output dictionary
    sentiments = {0: "Non Hopefull", 1: "Hopefull"}
    
    # show results
    result = {"prediction": sentiments[output], "Probability": output_probability}
    return result

#if __name__ == '__main__':
 #   uvicorn.run(app,host="127.0.0.1",port=8000)
    
#use this to run --  uvicorn main:app --reload

