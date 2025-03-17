from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Text, Dict, List
import requests
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

# Load KCC dataset
kcc_df = pd.read_csv("cleaned_kcc_dataset.csv")

# Preprocess dataset: Convert to lowercase and handle NaN values
kcc_df["querytext"] = kcc_df["querytext"].astype(str).str.lower().fillna("")
kcc_df["kccans"] = kcc_df["kccans"].astype(str).fillna("No information available.")

# TF-IDF Vectorizer for text similarity
vectorizer = TfidfVectorizer()
query_vectors = vectorizer.fit_transform(kcc_df["querytext"])


def find_best_match(user_query: str) -> str:
    """Finds the best matching answer from KCC dataset using TF-IDF and fuzzy matching."""

    # Convert user query to lowercase
    user_query = user_query.lower()

    # TF-IDF similarity check
    user_vector = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_vector, query_vectors)

    # Get the highest similarity score
    best_match_index = np.argmax(similarities)
    best_match_score = similarities[0, best_match_index]

    # If TF-IDF confidence is low, apply fuzzy matching
    if best_match_score < 0.5:
        max_fuzz_score = 0
        best_fuzz_match = None

        for idx, text in enumerate(kcc_df["querytext"]):
            fuzz_score = fuzz.partial_ratio(user_query, text)
            if fuzz_score > max_fuzz_score:
                max_fuzz_score = fuzz_score
                best_fuzz_match = idx

        if max_fuzz_score > 60:  # Adjust threshold as needed
            best_match_index = best_fuzz_match

    return kcc_df.iloc[best_match_index]["kccans"]


# Action for Agriculture Information
class ActionFetchAgricultureInfo(Action):
    def name(self) -> Text:
        return "action_fetch_agriculture_info"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[
        Dict[Text, Any]]:
        user_query = tracker.latest_message.get("text", "").lower().strip()

        if not user_query:
            dispatcher.utter_message(text="I couldn't understand your query. Please ask again.")
            return []

        answer = find_best_match(user_query)
        dispatcher.utter_message(text=answer)
        return []


class ActionFetchWeatherInfo(Action):
    def name(self) -> Text:
        return "action_fetch_weather_info"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[
        Dict[Text, Any]]:
        location = next(tracker.get_latest_entity_values("location"), None)

        if location:
            api_key = "5808653d0163d8f3058b9c5f37808d97"
            url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
            response = requests.get(url)
            data = response.json()

            if response.status_code == 200:
                temp = data['main']['temp']
                weather_desc = data['weather'][0]['description']
                message = f"The current temperature in {location} is {temp}°C with {weather_desc}."
            else:
                message = "Sorry, I couldn't fetch the weather details. Please try again later."
        else:
            message = "Please provide a location to fetch weather information."

        dispatcher.utter_message(text=message)
        return []


class ActionFetchHorticultureInfo(Action):
    def name(self) -> Text:
        return "action_fetch_horticulture_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        crop = next(tracker.get_latest_entity_values("fruit"), None) or next(
            tracker.get_latest_entity_values("vegetable"), None)
        if not crop:
            dispatcher.utter_message(text="Please specify a fruit or vegetable for horticulture information.")
            return []

        crop = crop.lower()
        user_query = tracker.latest_message.get("text").lower()  # Get full user query
        user_query = f"{crop} {user_query}"  # Append crop to the query

        answer = find_best_match(user_query)  # Find the best matching response
        if answer:
            dispatcher.utter_message(text=f"Horticulture information for {crop}: {answer}")
        else:
            dispatcher.utter_message(
                text=f"Sorry, I couldn't find specific information for '{user_query}'. Please try rephrasing your question.")
        return []


# Action for Crop Recommendation
class ActionFetchCropRecommendation(Action):
    def name(self) -> Text:
        return "action_fetch_crop_recommendation_info"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[
        Dict[Text, Any]]:
        user_query = tracker.latest_message.get("text", "").strip().lower()

        if not user_query:
            dispatcher.utter_message(text="I couldn't understand your query. Please ask again.")
            return []

        answer = find_best_match(user_query)
        dispatcher.utter_message(text=answer)
        return []


class ActionFetchSoilRecommendation(Action):
    def name(self) -> Text:
        return "action_fetch_soil_recommendation"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[
        Dict[Text, Any]]:
        user_query = tracker.latest_message.get("text", "").strip().lower()

        if not user_query:
            dispatcher.utter_message(text="I couldn't understand your query. Please ask again.")
            return []

        answer = find_best_match(user_query)
        dispatcher.utter_message(text=answer)
        return []
