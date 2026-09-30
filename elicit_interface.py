import os
import re
import time
from pathlib import Path
import requests
from dotenv import load_dotenv

def elicit_search(term: str, poll_seconds: int = 5, timeout: int = 1800) -> str:

    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.getenv("ELICIT_API_KEY")
    if not api_key:
        raise RuntimeError("ELICIT_API_KEY is missing from the environment or .env file")

    BASE = "https://elicit.com/api/v2/sessions/agents"
    HEADERS = {"Authorization": f"Bearer {api_key}"}
 


    research_question = f"""
    Give me 10 brief numbered sentences of broad breakthroughs for {term} research,
    development, and bringing to market for Department of Veterans Affairs associated
     work. Use review articles. Make the sentences brief and do not refer to the
    VA directly. Be broad with inception to current day including early research.
    """
   

    """Send one query to the Elicit Research Agent and return the answer text."""
    r = requests.post(BASE, headers=HEADERS, json={"query": research_question}, timeout=60)
    r.raise_for_status()
    session_id = r.json()["sessionId"]
 
    cursor, answer = None, None
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{BASE}/{session_id}/events", headers=HEADERS,
                         params={"cursor": cursor} if cursor else None, timeout=60)
        r.raise_for_status()
        page = r.json()
        cursor = page["cursor"]
 
        for ev in page["events"]:
            if ev["kind"] == "agent_message":
                answer = ev["text"]  # later snapshots replace earlier ones
            elif ev["kind"] == "question":
                answer = answer or ev["text"]  # agent needs input; return its question
 
        if page["status"] == "failed":
            raise RuntimeError(f"Elicit session {session_id} failed")
        if page["status"] == "pausedForInsufficientQuota":
            raise RuntimeError(f"Elicit session {session_id} paused: out of quota")
        if page["status"] == "completed" and answer:
            entries = []
            items = re.split(r"\b\d+\.\s*", answer)
            for item in items:
                entry = item.strip().capitalize()
                if entry:
                    entries.append(entry)
                if len(entries) == 10:
                    break
            return entries
        time.sleep(poll_seconds)
 
    raise TimeoutError(f"Elicit session {session_id} timed out")
