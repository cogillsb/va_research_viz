import os
import re
import time
from pathlib import Path
import requests
from dotenv import load_dotenv
import streamlit as st

def elicit_search(term):
    load_dotenv(Path(__file__).with_name(".env"))
    api_key = st.secrets["ELICIT_API_KEY"]
    if not api_key:
        raise RuntimeError("ELICIT_API_KEY is missing from the environment or .env file")

    research_question = f"""
        What are the 10 most significant breakthroughs in Department of Veterans Affairs associated research, development, "
        and clinical trials that brought {term} to market and shaped ongoing research? 
        Summarize them as a numbered list, not a timeline. Use only high-impact review 
        articles and provide a list of citations.
        """
   
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.post(
        "https://elicit.com/api/v1/reports",
        headers=headers,
        json={
            "researchQuestion": research_question,
            "maxSearchPapers": 50,
            "maxExtractPapers": 10,
        },
    )
    report_id = response.json()["reportId"]

    while True:
        time.sleep(30)
        report = requests.get(
            f"https://elicit.com/api/v1/reports/{report_id}",
            headers=headers,
        ).json()
        if report["status"] == "completed":
            summary = report["result"]["summary"]
            entries = []
            items = re.split(r"\(\d+\)", summary)[1:]
            for item in items:
                entry = item.strip().capitalize()
                if entry:
                    entries.append(entry)
                if len(entries) == 10:
                    break


            return entries


