import argparse
import base64
from datetime import datetime
from dotenv import load_dotenv
from enum import Enum
from groq import Groq
import io
import json
import logging
import matplotlib
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
import pandas
from pydantic import BaseModel, Field
import os
import re
import sqlite3
from typing import Union, Callable

from function_schema import sql_query_schema, generate_graph_schema, calculate_schema
from prompts import short_system_prompt as system_prompt

load_dotenv()

class ModelFamilies(Enum):
    LLAMA = "llama"
    QWEN = "qwen"
    MISTRAL = "mistral"

STOCK_QUESTIONS = [
    'How many rows are there?'
]
STOCK_HISTORY = [ {'role': 'user', 'content': question} for question in STOCK_QUESTIONS ]

MAX_HISTORY = 100
MAX_COMPLETION_TOKENS = 4096

env_endpoint_type = os.getenv("AI_ENDPOINT_TYPE","").strip().upper()
env_endpoint_id = os.getenv("AI_ENDPOINT_ID","")
env_model_name = os.getenv("MODEL_NAME","")

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Manages the RAG llama model"
    )
    parser.add_argument(
        "endpoint_id",
        nargs='?',
        help="The remote system endpoint id",
        default=env_endpoint_id
    )
    parser.add_argument(
        "-l", "--list",
        help="List the models supported by the endpoint",
        dest="list",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "-m", "--model",
        help="The model you want to use.  Call me with --list to see a list.",
        dest="model",
        default=env_model_name
    )
    parser.add_argument(
        "-t", "--type",
        help="The type of system we're talking to.",
        choices=["VLLM","GROQ","SGLANG","RUNPOD"],
        default=env_endpoint_type,
        dest="endpoint_type"
    )
    parser.add_argument(
        "-v", "--verbose",
        help="Print each message either received from or sent to the AI",
        default=False,
        dest="verbose",
        action="store_true"
    )
    args = parser.parse_args()
    
    if args.endpoint_type == "RUNPOD" and not args.endpoint:
        env_name="AI_ENDPOINT_ID"
        raise ValueError(
            f"Runpod endpoint ID required.  Provide it as an argument or set {env_name} in the environment."
        )

    return args

class Bcolors:
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    CYAN = '\033[96m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'

def log_interaction(message:dict):
    
    ef = f'{Bcolors.ENDC}'
    bf = Bcolors.BOLD
    qf = f'{bf}{Bcolors.CYAN}'
    af = f'{bf}{Bcolors.MAGENTA}'
    atf = qf

    print(f'{af}Role: {ef}{message['role']}')
    print(f'{atf}Content:{ef}\n{message['content']}')    
    print("".ljust(80, u'\u2550')) 

def log_ai_message(msg:dict):
    logging.info("".ljust(80, u'\u2550')) 
    logging.info(json.dumps(msg, indent=4))
    logging.info("".ljust(80, u'\u2550'))
    

class ToolDef(BaseModel):
    requires_followup: bool = Field(description="Whether to call the chat model again after tool results")
    requires_db_connection: bool = Field(description="Whether the function takes the db connection as the first argument.")
    fn: Callable = Field(description="A pointer to the function to call when directed by the model")

class ClientWrapper:
    def __init__(self, endpoint_type, endpoint_url, api_key, model=env_model_name):        
        self.client = (
            Groq(api_key=api_key) 
            if endpoint_type == "GROQ" else 
            OpenAI(base_url=endpoint_url, api_key=api_key)
        )
        self.model = model
        self.history = []
        self.max_history = MAX_HISTORY
        self.tool_schema = []
        self.tools:dict[str, ToolDef] = {}
        self.max_completion_tokens = MAX_COMPLETION_TOKENS
        self.models = []
        self.verbose = False
        self.system_logged = False

    def set_model(self, model:str):
        if model:
            self.model = model
        elif env_model_name:
            self.model = env_model_name
        else:
            # We weren't given a model and we don't 
            # have a default, so try seeing what the client
            # supports and picking one.
            self.model = self.default_model_from_list(
                self.model_names()
            )

    def model_names(self):
        if not self.models:
            self.models = self.client.models.list().data or []
        
        names = [ m.id for m in self.models ]
        names.sort()
        return names

    def default_model_from_list(self, names):
        def_model = None
        for model_name in names:
            if def_model is None: def_model = model_name
            if ("llama-3.3" in model_name or "llama-3.1" in model_name) and not "vision-" in model_name:
                def_model = model_name
                break
        
        return def_model
    
    def add_system(history):
        messages = [{
            "role": "system",
            "content": system_prompt
        }]

        messages.extend(history)
        return messages
    
    def compose_messages(self, new_message:Union[dict,ChatCompletionMessage]) -> list[dict]:
        
        del self.history[:max(0, len(self.history) - (self.max_history - 1))]
        
        ### model dump can stick keys with None values and this causes pydantic
        ### to throw errors.  We're going to drill any keys pointing to None.
        msg = new_message \
            if type(new_message) is dict else \
                new_message.model_dump(exclude_none=True)

        self.history.append(msg)
        log_ai_message(msg)
        messages = ClientWrapper.add_system(self.history)

        if self.verbose:
            if not self.system_logged:
                log_interaction(messages[0])
                self.system_logged = True
            log_interaction(messages[-1])

        return messages

    def handle_tools(self, tool_calls):

        # Process each tool call
        need_followup = False
        for tool_call in tool_calls:
            
            function_name = tool_call.function.name
            fdef = self.tools.get(function_name, None)
            if not fdef:
                messages = self.compose_messages({
                    'tool_call_id': tool_call.id,
                    'role': 'tool',
                    'name': function_name,
                    'content': json.dumps({'error': f"No registered function: '{function_name}'"})
                })
            else:
                need_followup = need_followup or fdef.requires_followup
                try:
                    args = json.loads(tool_call.function.arguments)
                    # Call the tool and get the response
                    function_response = fdef.fn(self.connection, **args) \
                        if fdef.requires_db_connection else \
                        fdef.fn(**args)
                except Exception as e:
                    need_followup = True
                    logging.error(e)
                    function_response = { "error": str(e) }

                messages = self.compose_messages({
                    "tool_call_id": tool_call.id, 
                    "role": "tool", # Indicates this message is from tool use
                    "name": function_name,
                    "content": json.dumps(function_response),
                })

        # Make an API call with the results of the tool calls
        if not need_followup:
            # This code will only return the results from the
            # last tool to the user.  If multiple tool calls get
            # made, none of which need followup, then the user only
            # sees the final result.  Hopefully, that's OK because
            # there really isn't a notion of sending multiple responses
            # to the user for a single query.
            last_msg = self.history[-1]
            last_msg['role'] = 'assistant'
            return ChatCompletionMessage(**last_msg)            

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        # Return the final response
        return response.choices[0].message


    def ask_question(self, question: str):
        new_message = {
            "role": "user",
            "content": question,
        }
        
        messages = self.compose_messages(new_message)

        # Make the initial API call to the AI endpoint
        response = self.client.chat.completions.create(
            model=self.model,          # LLM to use
            messages=messages,         # Conversation history
            stream=False,
            tools=self.tool_schema,    # Available tools (i.e. functions) for our LLM to use
            tool_choice="auto",        # Let our LLM decide when to use tools
            max_completion_tokens=self.max_completion_tokens # Maximum number of tokens to allow in our response
        )

        # Extract the response and any tool call responses
        response_message = response.choices[0].message
        messages = self.compose_messages(response_message)

        if ( tools := response_message.tool_calls ):
            response_message = self.handle_tools(tools)
            messages = self.compose_messages(response_message)
        
        # Return the result to the user
        return response_message.content

date_fix = re.compile(" \\([A-Z][a-zA-Z ]+ Time\\)")
def to_date(val):
    clean_date = date_fix.sub("",val)
    formats = [
        "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%a %b %d %Y %H:%M:%S %Z%z"
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(clean_date, fmt)
            return dt
        except ValueError:
            pass
    
    raise ValueError(f"No known date format for str '{val}")

def to_date_series(val):
    return val.apply(to_date)

def run_generate_graph(connection: sqlite3.Connection, code: str) -> str:
    """
    Executes a block of python code that is expected to use matplotlib to generate a graph
    and then returns an svg tag that can be displayed to the user.

    Args:
        code: The block of python code that can generate an image

    Returns:
        str: A markdown, embeddable SVG
    """

    
    res = "Something went wrong.  Please try again."
    try:
        # There is a sucky thing where, for some reason, the model wants to call show off of
        # a panda dataframe bar code.  This doesn't work because the call returns an Axis, not
        # a plot and show can't be shown.  SO... we're going to look for ".show()" at the end
        # and kill it if we see it.  Yes, a hack.  Maybe the smarter models won't do this and maybe
        # the code specific model can be convinced to do it.
        clean = re.sub("\\.show\\(\\)(\n?)$", "\\1", code.strip())

        ### This version of sql_query gets put in "locals" so that
        ### if the generated code references a graph, we have it.
        def sql_query(*a, **kw) -> dict:
            args = a if a else list(kw.values())
            query = args[0] if args else ''
            if not query: raise ValueError("Can't figure out the query")
            return run_sql_query(connection, query)
        
        loc = dict(locals())
        exec(clean, globals(), loc)
        if (plt := loc.get('plt')):
            # Save the plot to a BytesIO object in SVG format
            svg_buffer = io.BytesIO()
            #plt.tight_layout()
            plt.gcf().set_size_inches(10, 10)
            plt.savefig(svg_buffer, format='svg', dpi=100)
            # Get the string representation of the SVG
            svg_string = svg_buffer.getvalue().decode()        
            res = f'![svg image](data:image/svg+xml;base64,{base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')})'
        else:
            logging.warning("Didn't get a plt object from image.")

    except Exception as e:
        msg = str(e) + "\n==========\n" + clean + "\n===========" 
        logging.error(msg)
    
    return res
    

def get_connection(project = 'phil') -> sqlite3.Connection:
    return sqlite3.connect(project + '.db')

def initialize_database(
      url: str = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSHkGmbCEOqcLxGusQahQ0yziElavRjcTIMApeyffC80fgG0fYaKHs_1-GxzvxYN2HDFtqNfLuPv7ep/pub?gid=0&single=true&output=csv',
      datecolumns = ['Submission_time', 'Date_Observation'],
      project = 'phil',
      table = 'eval'
) -> sqlite3.Connection:
    dataframe = pandas.read_csv(url, index_col=0)
    to_replace = {}
    for col in dataframe.columns:
        new_col = col.strip().replace(' ', '_').replace(':','').replace('/','_')
        if new_col != col: to_replace[col] = new_col
    if to_replace: dataframe.rename(columns=to_replace, inplace=True)
    if datecolumns: dataframe[datecolumns] = dataframe[datecolumns].apply(to_date_series)
    connection = sqlite3.connect(project + '.db')
    dataframe.to_sql(table, connection, if_exists='replace')
    return connection

def run_sql_query(connection: sqlite3.Connection, query: str) -> dict:
    """
    Runs an SQL query and returns the result as a dict

    Args:
        connection: The database connection to use for the query
        query: The query to be run

    Returns:
        dict: The results object
    """
    result = None
    
    def do_query(q):
        try:
            result = pandas.read_sql_query(q, connection).to_dict(orient='records')
            return (result, False)
        except Exception as e:
            logging.warning(f"Bad query: '{query}'")
            logging.warning(f'Error: {e}')
            return ({"error": str(e)}, True)

    result, errored = do_query(query)
    if errored and query.endswith(")"):
        ## I've seen cases where a trailing ")" gets added to the query
        ## and there is no open "(".  If we fail and the query ends with a ")", I'm
        ## going to remove it and try again.
        result, errored = do_query(query[-1])

    return result
    
re_python_graph = re.compile("```python\n(.*import matplotlib\\.pyplot as plt\\s.*)```",re.DOTALL)
def convert_python_if_necessary(connection: sqlite3.Connection, message: str):
    """
    Looks for strings that intend to produce charts and redirects
    to charting code here.  I REALLY wanted to be fancy and get the
    agent to call my code automatically but sometimes it is refusing to do it.
    SO...

    Args:
        connection: The active database connection
        message: The message returned from the LLM

    Returns:
        str: Either the message unchanged or the code run and
             converted into svg
    """
    
    if (code := re_python_graph.search(message)):
        print(code.group(1))
        img = run_generate_graph(connection, code.group(1))
        return re_python_graph.sub(img, message)
 
    return message

def family_from_str(model:str) -> ModelFamilies:
    low = model.lower()
    if "/qwen2." in low or low.startswith("qwen2."): return ModelFamilies.QWEN
    if "/mistral-" in low or low.startswith("mistral-"): return ModelFamilies.MISTRAL
    if "/meta-" in low or "-llama-" or low.startswith("llama-"): return ModelFamilies.LLAMA

qwen_re = re.compile("^\\<tool_call\\>\n(.*)\n\\<\\/tool_call\\>")
def parse_tool_calls(model_family: ModelFamilies, completion: ChatCompletion) -> ChatCompletion:

    message:ChatCompletionMessage = completion.choices[0].message
    # If we've already got parsed tool calls, leave the message along.
    if message.tool_calls: return completion

    # If the model isn't of of the ones we support, return the message untouched.
    # Right now, we only support parsing Qwen.

    if not model_family == ModelFamilies.QWEN:
        return completion

    re = qwen_re
    #OK, qwen tool calls look <tool_call>\n{json stuff}\n</tool_call>
    # This code won't work if there is more than one tool call in the content.
    # I've never seen a string like that, so I don't know exactly how to 
    # change the regex.
    c2 = match.groups(0)[0] if (match := re.match(message.content)) else ""
    if not c2: return completion
    try:
        call = json.loads(c2)
        # In OpenAI land, arguments are a string, not a json obj - WHICH SUCKS
        call['arguments'] = json.dumps(call['arguments'])
        tool_call = ChatCompletionMessageToolCall(**{'id': '123', 'function': call, 'type': 'function'})
        message.tool_calls = [tool_call]
    except Exception:
        # OK, it's not json, so....
        pass
    
    return completion

def calculate(expression):
    """Evaluate a mathematical expression"""
    result = eval(expression)
    return {"result": result}

def get_endpoint_url(endpoint_type, endpoint_id):

    if endpoint_type == "VLLM": endpoint_url = "http://localhost:8000/v1"
    elif endpoint_type == "SGLANG": endpoint_url = "http://localhost:30000/v1"
    elif endpoint_type == "RUNPOD": endpoint_url = f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"
    else: endpoint_url = endpoint_type             

    return endpoint_url

def get_client(

    db_connection:sqlite3.Connection, 
    model:str = env_model_name, 
    history:list[dict] = [],
    endpoint_type = env_endpoint_type, endpoint_id = env_endpoint_id):
    endpoint_url = get_endpoint_url(endpoint_type, endpoint_id)

    api_key = (
        os.getenv("GROQ_API_KEY")
        if endpoint_type == "GROQ" else
        os.getenv("RUNPOD_API_KEY")
    )

    client = ClientWrapper(endpoint_type, endpoint_url, api_key)

    client.tool_schema = [
        calculate_schema,
        sql_query_schema,
        generate_graph_schema
    ]

    client.tools = {
        'sql_query': ToolDef(
            requires_followup=True,
            requires_db_connection=True,
            fn=run_sql_query 
        ),
        'generate_graph': ToolDef(
            requires_followup=False,
            requires_db_connection=True,
            fn=run_generate_graph
        ),
        'calculate': ToolDef(
            requires_db_connection=False,
            requires_followup=True,
            fn=calculate
        )
    }

    client.set_model(model)
    client.history = history

    client.connection = db_connection

    return client


if __name__ == '__main__':

    args = get_arguments()
    client = get_client(
        initialize_database(), 
        args.model, 
        [], 
        args.endpoint_type, 
        args.endpoint_id
    )

    if args.list:
        for name in client.model_names():
            print(name)
        exit()
    

    ef = f'{Bcolors.ENDC}'
    bf = Bcolors.BOLD
    qf = f'{bf}{Bcolors.CYAN}'
    af = f'{bf}{Bcolors.MAGENTA}'
    qtf = Bcolors.WHITE
    atf = ''

    print(f"Using model: '{client.model}'")

    client.verbose = args.verbose

    for q in STOCK_QUESTIONS:
        print(f'{qf}Asked:{ef} {qtf}{q}{ef}')
        answer = client.ask_question('How many rows are there?')
        print(f'{af}Answered:{ef} {atf}{answer}{ef}')
        print('')

    text ="x"
    while text:
        text = input(f'{qf}Ask:{ef} {qtf}')
        print(f'{ef}', end='')
        if not text: continue
        response = client.ask_question(text)
        print(f'{af}Answer:{ef} {atf}{response}{ef}')
        print('')

    print('Goodbye.')

