import requests as r

# add review
text = "Everything which is against nature and unnatural is legalized in US in the name of so called choice. Freedom and love.. LGBTQ is its perfect example.. LGBTQ is not a choice or love its a mental sickness which needs proper treatment "

keys = {"text": text}

prediction = r.get("http://127.0.0.1:8000/predict/", params=keys)

results = prediction.json()
print(results["prediction"])
print(results["Probability"])