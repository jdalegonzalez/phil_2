import argparse
import base64
from datetime import datetime
from enum import Enum
import io
import json
import logging

from groq import Groq

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

import matplotlib
import pandas
import os
import re
import sqlite3

from prompts import short_system_prompt as system_prompt
from function_schema import sql_query_schema, generate_graph_schema, calculate_schema

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

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Manages the RAG llama model"
    )
    parser.add_argument(
        "endpoint",
        nargs='?',
        help="The runpod endpoint id",
        default=""
    )
    parser.add_argument(
        "-g", "--groq",
        help="Use the groq endpoint.",
        dest="groq",
        default=False,
        action='store_true'
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
        default=""
    )
    parser.add_argument(
        "-s", "--sql", 
        help="Use a locally running sgl-lang (https://docs.sglang.ai/start/install.html) server instead of runpod",
        dest='sglang',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "-v", "--vllm", 
        help="Use a locally running vLLM (https://docs.vllm.ai/en/stable/getting_started/installation/index.html) server instead of runpod",
        dest='vllm',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--verbose",
        help="Print each message either received from or sent to the AI",
        default=False,
        dest="verbose",
        action="store_true"
    )
    args = parser.parse_args()
    
    if not args.endpoint:
        args.endpoint = os.environ.get("RUNPOD_ENDPOINT_ID")
    
    if not args.vllm and not args.sglang and not args.groq and not args.endpoint:
        name="Runpod"
        env_name="RUNPOD_ENDPOINT_ID"
        raise ValueError(f"{name} endpoint ID required if you're not running locally.  Either provide it as an argument or set {env_name} in the environment.")

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

def log_interaction(message):
    
    to_log = message.model_dump() if not type(message) is dict else message

    ef = f'{Bcolors.ENDC}'
    bf = Bcolors.BOLD
    qf = f'{bf}{Bcolors.CYAN}'
    af = f'{bf}{Bcolors.MAGENTA}'
    atf = qf

    print(f'{af}Role: {ef}{to_log['role']}')
    print(f'{atf}Content:{ef}\n{to_log['content']}')    
    print("".ljust(80, u'\u2550')) 

class ClientWrapper:
    def __init__(self, use_groq, api_key):        
        self.client = (
            Groq(api_key=api_key) 
            if use_groq else 
            OpenAI(base_url=endpoint_url, api_key=api_key)
        )
        self.model = os.getenv('MODEL_NAME')
        self.history = []
        self.max_history = MAX_HISTORY
        self.tool_schema = []
        self.tools = []
        self.max_completion_tokens = MAX_COMPLETION_TOKENS
        self.models = []
        self.verbose = False
        self.system_logged = False

    def set_model(self, model:str):
        if model:
            self.model = model
        elif (os_model := os.getenv('MODEL_NAME')):
            self.model = os_model
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
    
    def compose_messages(self, new_message):
        del self.history[:max(0, len(self.history) - (self.max_history - 1))]
        self.history.append(new_message)
        messages = ClientWrapper.add_system(self.history)

        if self.verbose:
            if not self.system_logged:
                log_interaction(messages[0])
                self.system_logged = True
            log_interaction(messages[-1])

        return messages

    def handle_tools(self, tool_calls):

        # Process each tool call
        for tool_call in tool_calls:
            
            # TODO: Add error handling here
            function_name = tool_call.function.name
            f = self.tools[function_name]
            args = json.loads(tool_call.function.arguments)
            
            # Call the tool and get the response
            function_response = f(**args)
            messages = self.compose_messages({
                "tool_call_id": tool_call.id, 
                "role": "tool", # Indicates this message is from tool use
                "name": function_name,
                "content": json.dumps(function_response),
            })

        # Make an API call with the results of the tool calls
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
        print("*****\n",clean)
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
    try:
        result = pandas.read_sql_query(query, connection).to_dict(orient='records')
    except Exception as e:
        logging.warning(f"Bad query: '{query}'")
        logging.warning(f'Error: {e}')
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
        print("**** GOTCHA*****")
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

def clean_message(response):
    """ Apparently, sometimes we get back a response that isn't valid
        as a part of the history.  SO... we need to clean it up.  There
        might be a better way to do this.  I need to check the Groq doc
    """
    new_message = response.message.model_dump()
    if new_message.get('function_call', None) is None:
        new_message.pop('function_call',None)
    
    new_message.pop('reasoning', None)
    
    return new_message


def get_answer(
    connection:sqlite3.Connection, 
    question:str, history: list[dict] = [],
    model: str = 'llama', final_model: str = None) -> tuple[str, list[dict]]:

    model_family = family_from_str(model)

    def sql_query(query: str):
        """Run a SQL SELECT query on a SQLite database and return the results."""
        return run_sql_query(connection, query)
   
    def generate_graph(code: str):
        """
        Executes a block of python code to generate charts and graphs.  The 
        code is expected to use matplotlib.pyplot to generate the graph and
        use data already fetched or call the sql_query function

        Args:
            code: The python code to generate the graph or chart

        Returns:
            str: an embeddale svg.
        """
        return run_generate_graph(connection, code)
   
    available_functions = {
        'sql_query': {'func': sql_query, 'needs_followup': True },
        'generate_graph': {'func': generate_graph, 'needs_followup': False}
    }

    history.append({'role': 'user', 'content': question})
    
    completion = client.chat.completions.create(
        model=model,
        messages=history,
        stream=False,
        tools=[sql_query_schema,generate_graph_schema],
        tool_choice="auto",
        temperature=1,
        max_completion_tokens=4096
    )
    
    parse_tool_calls(model_family, completion)
    response = completion.choices[0]
    print("**** FIRST PASS ****")
    print(response)
    print("*********************")
    # TODO: The move here might be to first call a code
    # generating bot.  Maybe I can tell the bot that
    # if the request doesn't require code, it should
    # respond with "Not for me" or something.  Speed
    # is my biggest concern
    if response.message.tool_calls:
        # There may be multiple tool calls in the response
        new_message = clean_message(response)
        
        print("==*+*++*=",new_message,"==*+*+*++==")

        history.append(new_message)
        for tool in response.message.tool_calls:
            # Ensure the function is available, and then call it
            if function_info := available_functions.get(tool.function.name):
                logging.debug(f"Calling function: '{tool.function.name}'")
                logging.debug(f"Arguments:{tool.function.arguments}'")
                print(f"Calling function: '{tool.function.name}'")
                print(f"Arguments:{tool.function.arguments}'")
                args = json.loads(tool.function.arguments)
                output = function_info['func'](**args)
                history.append({'role': 'tool', 'tool_call_id': tool.id, 'content': str(output), 'name': tool.function.name})
            else:
                logging.warning(f"Function '{tool.function.name}' not found")
        
        # This code has a problem in that it will only give the user the very last response.
        # Hopefully, that's OK? Should this code handle the possibility that a response
        # to a tool calls chat ALSO needs calls?
        last_name = (history[-1] or {}).get('name','')
        last_info = available_functions.get(last_name,{'func': None, 'needs_followup': True})
        followup = last_info['needs_followup']
    
        if followup:
            if final_model is None: final_model = model
            print("****** SECOND PASS *******")
            for h in history: print("---",h,"---")
            completion = client.chat.completions.create(
                model=final_model,
                messages=history,
            )
            response = completion.choices[0]
            print(response)
            print("***************************")
        else:
            last_msg = history[-1]
            last_msg['role'] = 'assistant'
            response.message = ChatCompletionMessage(**last_msg)

    response.message.content = convert_python_if_necessary(connection, response.message.content)
    history.append(response.message.model_dump())
    
    return (response.message.content, history)

def calculate(expression):
    """Evaluate a mathematical expression"""
    result = eval(expression)
    return {"result": result}

def groq_test(connection, user_prompt, model):

    def sql_query(query): return run_sql_query(connection, query)
    
    # Initialize the conversation with system and user messages
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    # Define the available tools (i.e. functions) for our model to use
    tools = [ calculate_schema, sql_query_schema ]

    # Make the initial API call to Groq
    response = client.chat.completions.create(
        model=model, # LLM to use
        messages=messages, # Conversation history
        stream=False,
        tools=tools, # Available tools (i.e. functions) for our LLM to use
        tool_choice="auto", # Let our LLM decide when to use tools
        max_completion_tokens=4096 # Maximum number of tokens to allow in our response
    )

    # Extract the response and any tool call responses
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    if tool_calls:
        # Define the available tools that can be called by the LLM
        available_functions = {
            "calculate": calculate,
            "sql_query": sql_query,
        }
        # Add the LLM's response to the conversation
        messages.append(response_message)

        # Process each tool call
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            f = available_functions[function_name]
            args = json.loads(tool_call.function.arguments)
            # Call the tool and get the response
            function_response = f(**args)
            # Add the tool response to the conversation
            messages.append(
                {
                    "tool_call_id": tool_call.id, 
                    "role": "tool", # Indicates this message is from tool use
                    "name": function_name,
                    "content": json.dumps(function_response),
                }
            )
        
        # Make a second API call with the updated conversation
        second_response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        # Return the final response
        return second_response.choices[0].message.content

args = get_arguments()
endpoint_id = args.endpoint

if args.vllm:     endpoint_url = "http://localhost:8000/v1"
elif args.sglang: endpoint_url = "http://localhost:30000/v1"
else:             endpoint_url = f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"

api_key = (
    os.environ.get("GROQ_API_KEY")
    if args.groq else
    os.environ.get("RUNPOD_API_KEY")
)

client = ClientWrapper(args.groq, api_key)
client.tool_schema = [ calculate_schema, sql_query_schema ]

if __name__ == '__main__':

    if args.list:
        for name in client.model_names():
            print(name)
        exit()
    
    client.set_model(args.model)

    ef = f'{Bcolors.ENDC}'
    bf = Bcolors.BOLD
    qf = f'{bf}{Bcolors.CYAN}'
    af = f'{bf}{Bcolors.MAGENTA}'
    qtf = Bcolors.WHITE
    atf = ''

    print(f"Using model: '{client.model}'")

    connection = initialize_database()

    def sql_query(query:str) -> dict:
        return run_sql_query(connection, query)

    def generate_graph(code: str) -> str:
        return run_generate_graph(connection, code)    
    
    client.tools = {
        'sql_query': sql_query,
        'generate_graph': generate_graph,
        'calculate': calculate
    }
    
    client.verbose = args.verbose
    #print(client.ask_question("What is 24 * 10 + 15"))
    #print(client.ask_question("How many rows are in the database"))
    #print(client.ask_question("What are the names of the columns"))

    #exit(1)

    for q in STOCK_QUESTIONS:
        print(f'{qf}Asked:{ef} {qtf}{q}{ef}')
        response, history = get_answer(connection, 'How many rows are there?', history=history, model=model)
        print(f'{af}Answered:{ef} {atf}{response}{ef}')
        print('')

    text ="x"
    while text:
        text = input(f'{qf}Ask:{ef} {qtf}')
        print(f'{ef}', end='')
        if not text: continue
        response, history = get_answer(connection, text, history=history)
        print(f'{af}Answer:{ef} {atf}{response}{ef}')
        print('')

    print('Goodbye.')

