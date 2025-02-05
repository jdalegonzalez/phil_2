from typing import Annotated

import os
import uuid
import logging

from sqlite3 import Connection

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request

from agent import get_connection, get_answer, initialize_database, STOCK_HISTORY


######## Constants
SESSION_AGE = 60 * 60 # 60 seconds * 60 = 1 hour
########

app = FastAPI()
key = os.getenv('TBOXAI_SESSION_KEY',str(uuid.uuid4()))

# Configure session middleware (using Redis for example)
# We'll want to change this to redis once we have more than 
# one server.
app.add_middleware(SessionMiddleware, secret_key=key, max_age=SESSION_AGE)

origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://localhost",
    "https://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "https://127.0.0.1",
    "https://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_session(request: Request):
    print("***")
    print(request.cookies)
    print("***")
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
    print("History", len(history))
    result, history = get_answer(connect_to_data(session), question, history)
    save_history(history, session)
    return result

@app.get("/")
async def read_root(q: str, session: SessionDep):
    result = ask_question(q, session)
    return {"answer": result}
