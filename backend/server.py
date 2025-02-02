from typing import Annotated

import os
import uuid
import logging

from fastapi import FastAPI, Depends
from agent import get_connection, get_answer, initialize_database
from sqlite3 import Connection

from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request

######## Constants
SESSION_AGE = 60 * 60 # 60 seconds * 60 = 1 hour
########

app = FastAPI()
key = os.getenv('TBOXAI_SESSION_KEY',str(uuid.uuid4()))

# Configure session middleware (using Redis for example)
# We'll want to change this to redis once we have more than 
# one server.
app.add_middleware(SessionMiddleware, secret_key=key, max_age=SESSION_AGE)

async def get_session(request: Request):
    return request.session

SessionDep = Annotated[dict, Depends(get_session)]

def connect_to_data(session: dict) -> Connection:
    # Ultimately, we'll determine if we
    # need to initialize the database - because
    # it's gone stale, or if we just need to
    # connect to an existing one - because 
    # we're in the same session.
    if not session.get('db_initialized', False):
        # We'll init the db once per session
        logging.info("Initializing the DB")
        connection = initialize_database()
        session['db_initialized'] = True
    else:
        logging.info("Reusing existing connection.")
        connection = get_connection()

    return connection

def load_history(session:dict):
    return session.get('history', [])

def save_history(history:list[dict], session:dict):
    session['history'] = history

def ask_question(question: str, session: dict) -> str:
    history = load_history(session)
    result, history, _ = get_answer(connect_to_data(session), question, history)
    save_history(history, session)
    return result

@app.get("/")
async def read_root(q: str, session: SessionDep):
    print(session)
    result = ask_question(q, session)
    return {"answer": result}
