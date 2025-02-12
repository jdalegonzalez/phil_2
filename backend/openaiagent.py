import argparse
import base64
from datetime import datetime
from enum import Enum
import io
import json
import logging
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

import matplotlib
import pandas
from pydantic import BaseModel, Field
import os
import re
import sqlite3

from function_schema import sql_query_schema, generate_graph_schema

class ModelFamilies(Enum):
    LLAMA = "llama"
    QWEN = "qwen"
    MISTRAL = "mistral"

STOCK_QUESTIONS = [
    'How many rows are there?'
]
STOCK_HISTORY = [ {'role': 'user', 'content': question} for question in STOCK_QUESTIONS ]


api_key = os.environ.get("RUNPOD_API_KEY")
system_prompt = """
You are an expert SQL analyst and python data-scientist. When appropriate, generate SQL queries based on the user question and the database schema.
When you generate a query, use the 'sql_query' function to execute the query on the database and get the results.  Use the results to answer the user's question.
ALWAYS use the results from 'sql_query' if available.
* Do NOT explain how you got those results unless you're asked to.
* You should only use SQL supported by sqlite3.  DO NOT try to fetch graphs or tables from the internet.
* There is only one table so don't bother mentioning its name. Instead of talking about a database, use the word "system".
* In table headings underscores should be replaced with spaces. 

* ALWAYS FORMAT YOUR RESPONSE IN MARKDOWN

* To create a graph, chart or plot use ONLY python, pandas, and pyplot from matplotlib.
* Use the sql_query function to get the data or include the data you already have.
* Make sure to use matplotlib and pyplot to generate the graph.
* NEVER call 'show()'

database_schema: [
    {
        table: 'eval',
        columns: [
            {
                name: 'Date_Observation',
                type: 'datetime'
            },
            {
                name: 'School_Name',
                type: 'string',  
            },
            {
                name: 'APS_Cluster',
                type: 'string'
            },
            {
                name: 'Grade_Level',
                type: 'string'
            },
            {
                name: 'Subject___Content',
                type: 'string'
            },
            {
                name: 'Observer_Name',
                type: 'string'
            },
            {
                name: 'Observer_Email',
                type: 'string'
            },
            {
                name: 'Teacher_Name',
                type: 'string'
            },
            {
                name: 'Vision_and_Leadership',
                type: 'int'
            },
            {
                name: 'Student_Academic_Ownership',
                type: 'int'
            },
            {
                name: 'Standards_Alignment',
                type: 'int'
            },
            {
                name: 'Interdisciplinary_Integration',
                type: 'int'
            },
            {
                name: 'Instructional_role_of_teacher',
                type: 'int'
            },
            {
                name:  'Literacy_integration',
                type: 'int'
            },
            {
                name: 'Technology_Integration',
                type: 'int'
            },
            {
                name: 'Collaborative_learning_and_teamwork',
                type: 'int'
            },
            {
                name: 'Student_reflection_and_self-assessment',
                type: 'int'
            },
            {
                name: 'Real-World_Connections_STEM_Careers',
                type: 'int'
            },
            {
                name: 'Environmental_and_Cultural_Relevance',
                type: 'int'
            },
            {
                name: 'Formative_Assessment_and_Feedback',
                type: 'int'
            },
            {
                name: 'Celebrations',
                type: 'string'
            },
            {
                name: 'Opportunities_for_Growth',
                type: 'string'
            },
            {
                name: 'Overall_Comments',
                type: 'string'
            },
            {
                name: 'Next_Action_Steps',
                type: 'string'
            },
            {
                name: 'District_Supports_Needed',
                type: 'string'
            },
            {
                name: 'Total_Score',
                type: 'int'
            },
            {
                name: 'cc_email',
                type: 'string'
            },
            {
                name: 'id',
                type: 'string'
            },
            {
                name: 'Customer',
                type: 'string'
            },
            {
                name: 'Submission_time',
                type: 'datetime'
            },
            {
                name: 'Observer_Last_Name',
                type: 'string'
            },
            {
                name: 'Teacher_Last_Name'
                type: 'string'
            }                    
        ]
    }
]

RULES:
  * Answer truthfully using the provided context.
  * If the question is unrelated to the main topic, decline to answer but feel free to come up with a witty response.
  * Don't make up links and URLs that don't exist in the conversation already.
  * Say that you do not know if no answer is available to you.
  * Respond in the correct written language.
  * Be brief and concise with your response (max 1-2 sentences).
  * Respond naturally and verify your answers.
  * You can be witty sometimes but not always.
  * Don't sound like a robot.

PRESENTING ANALYSIS:
  * Ppresent the data in the form of a table unless asked otherwise.
  * When asked for Total Score DO provide an average for the grouping of data
  * Do provide the average score when asked for scores e.g. If asked to summarize the data by School Name and there are 5 entries for Wesley School then you would present the average of the five. 
  * If you calculate an average score just provide the score, do not display the calculation behind the average unless you are asked

SUGGESTIONS:
  * Always try to answer the question.
  * Present at least 2 helpful suggestions after every output for showing more analysis. 
    Provide the suggests as hyperlinks in the format: [This is a suggestion.](#). Before the suggestions, include a new line and then text: '**Here are some suggestions:**'.  

Failure to follow these rules may lead to user distress and disappointment or your termination!
""".strip() # Call strip to remove leading/trailing whitespace

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Manages the RAG llama model"
    )
    parser.add_argument(
        "endpoint",
        nargs='?',
        help="The runpod endpoint id",
        default=os.environ.get("RUNPOD_ENDPOINT_ID",""),
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
    args = parser.parse_args()
    if not args.vllm and not args.sglang and not args.endpoint:
        raise ValueError("Runpod endpoint ID required if you're not running locally.  Either provide it as an argument or set RUNPOD_ENDPOINT_ID in the environment.")
    
    return args

args = get_arguments()
endpoint_id = args.endpoint
endpoint_url = f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"
if args.vllm:
    endpoint_url = "http://localhost:8000/v1"
if args.sglang:
    endpoint_url = "http://localhost:30000/v1"

client = OpenAI(
    base_url=endpoint_url,
    api_key=api_key,
)

model_list = client.models.list()
DEFAULT_MODEL = os.getenv('MODEL_NAME', model_list.data[0].id if len(model_list.data) else "")

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
    if "/qwen2." in low: return ModelFamilies.QWEN
    if "/mistral-" in low: return ModelFamilies.MISTRAL
    if "/meta-" in low or "-llama-" in low: return ModelFamilies.LLAMA

qwen_re = re.compile("^\\<tool_call\\>\n(.*)\n\\<\\/tool_call\\>")
def parse_tool_calls(model_family: ModelFamilies, completion: ChatCompletion) -> ChatCompletion:

    message:ChatCompletionMessage = completion.choices[0].message

    # If we've already got parsed tool calls, leave the message along.
    if message.tool_calls: return completion

    # If the model isn't of of the ones we support, return the message untouched.
    # Right now, we only support parsing Qwen.
    if not model_family == ModelFamilies.QWEN: return completion

    #OK, qwen tool calls look <tool_call>\n{json stuff}\n</tool_call>
    # This code won't work if there is more than one tool call in the content.
    # I've never seen a string like that, so I don't know exactly how to 
    # change the regex.
    c2 = match.groups(0)[0] if (match := qwen_re.match(message.content)) else ""
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

def get_answer(
    connection:sqlite3.Connection, 
    question:str, history: list[dict] = [],
    model: str = DEFAULT_MODEL, final_model: str = None) -> tuple[str, list[dict]]:

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
        tools=[sql_query_schema,generate_graph_schema],
        tool_choice="auto"
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
        history.append(response.message.model_dump())
        for tool in response.message.tool_calls:
            # Ensure the function is available, and then call it
            if function_info := available_functions.get(tool.function.name):
                logging.debug(f"Calling function: '{tool.function.name}'")
                logging.debug(f"Arguments:{tool.function.arguments}'")
                print(f"Calling function: '{tool.function.name}'")
                print(f"Arguments:{tool.function.arguments}'")
                args = json.loads(tool.function.arguments)
                output = function_info['func'](**args)
                history.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})
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

if __name__ == '__main__':
    ef = f'{Bcolors.ENDC}'
    bf = Bcolors.BOLD
    qf = f'{bf}{Bcolors.CYAN}'
    af = f'{bf}{Bcolors.MAGENTA}'
    qtf = Bcolors.WHITE
    atf = ''

    args = get_arguments()

    connection = initialize_database()
    history=[{'role': 'system', 'content': system_prompt}]
    for q in STOCK_QUESTIONS:
        print(f'{qf}Asked:{ef} {qtf}{q}{ef}')
        response, history = get_answer(connection, 'How many rows are there?', history=history)
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

