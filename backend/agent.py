import pandas as pd
import re
from datetime import datetime
import sqlite3
import argparse
import logging

from ollama import Client, chat, ChatResponse

DEFAULT_MODEL = 'aps'
DEFAULT_BASE  = 'llama3.1:8b'

system_prompt = """
You are an expert SQL analyst. When appropriate, generate SQL queries based on the user question and the database schema.
When you generate a query, use the 'sql_query' function to execute the query on the database and get the results.
Then, use the results to answer the user's question. You should only use SQL supported by sqlite3

Use javascript and the D3.js library to generate charts and graphs.

There is only one table so don't bother mentioning its name. Instead of talking about a database, use the word system.

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
    * Do present the data in the form a table unless asked otherwise.
    * When asked for Total Score DO provide an average for the grouping of data
    * Do provide the average score when asked for scores e.g. If asked to summarize the data by School Name and there are 5 entries for Wesley School then you would present the average of the five. 
    * If you calculate an average scores just provide the score, do not display the calculation behind the average unless you are asked

SUGGESTIONS:
    * Do present at least 2 helpful suggestions after every output for showing more analysis. Show as a button [suggestion 1](), [suggestion 2]() etc.

Failure to follow these rules may lead to user distress and disappointment or your termination!
""".strip() # Call strip to remove leading/trailing whitespace

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

def generate_graph(val):
    # TODO: Implement support for graph generation - somehow
    pass

def get_connection(project = 'phil') -> sqlite3.Connection:
    return sqlite3.connect(project + '.db')

def initialize_database(
      url: str = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSHkGmbCEOqcLxGusQahQ0yziElavRjcTIMApeyffC80fgG0fYaKHs_1-GxzvxYN2HDFtqNfLuPv7ep/pub?gid=0&single=true&output=csv',
      datecolumns = ['Submission_time', 'Date_Observation'],
      project = 'phil',
      table = 'eval'
) -> sqlite3.Connection:
    dataframe = pd.read_csv(url, index_col=0)
    to_replace = {}
    for col in dataframe.columns:
        new_col = col.strip().replace(' ', '_').replace(':','').replace('/','_')
        if new_col != col: to_replace[col] = new_col
    if to_replace: dataframe.rename(columns=to_replace, inplace=True)
    if datecolumns: dataframe[datecolumns] = dataframe[datecolumns].apply(to_date_series)
    connection = sqlite3.connect(project + '.db')
    dataframe.to_sql(table, connection, if_exists='replace')
    return connection

def run_sql_query(connection, query: str):
    """Run a SQL SELECT query on a SQLite database and return the results."""
    result = "Something went wrong"
    try:
        result = pd.read_sql_query(query, connection).to_dict(orient='records')
    except Exception as e:
        logging.warning(f"Bad query: '{query}'")
        logging.warning(f'Error: {e}')
    return result

# messages = [{'role': 'system', 'content': 'What is three plus one?'}]

def get_answer(
    connection:sqlite3.Connection, 
    question:str, history: list[dict] = [], 
    model: str = DEFAULT_MODEL, final_model: str = None) -> tuple[str, list[dict], ChatResponse]:

    if final_model is None: final_model = model
    def sql_query(query: str):
        return run_sql_query(connection, query)
   
    history.append({'role': 'user', 'content': question})
    available_functions = {
        'sql_query': sql_query,
        'generate_graph': generate_graph,
    }
    response = chat(
        model=model,
        messages=history,
        tools=[sql_query, generate_graph],
        format="json"
    )

    if response.message.tool_calls:
        # There may be multiple tool calls in the response
        for tool in response.message.tool_calls:
            # Ensure the function is available, and then call it
            if function_to_call := available_functions.get(tool.function.name):
                logging.debug(f"Calling function: '{tool.function.name}'")
                logging.debug(f"Arguments:{tool.function.arguments}'")
                output = function_to_call(**tool.function.arguments)
                logging.debug(f"****Function output:\n{output}\n****")
            else:
                logging.warning(f"Function '{tool.function.name}' not found")

        history.append(response.message)
        history.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})
        response = chat(final_model, messages=history)

    history.append({'role': 'assistant', 'content': response.message.content})

    # Now that we've got our answer, we're going to pull the "Message" out of the 
    # history because it's not serializable (and hopefully not necessary?)
    history  = [ itm for itm in history if type(itm) == dict ]
    return (response.message.content, history, response)

def create_model_with_system(model:str = DEFAULT_MODEL, from_model:str = DEFAULT_BASE):
    client = Client()
    result = client.create(
       model=model,
       from_=from_model,
       system=system_prompt,
       stream=False
    )
    return result

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Manages the RAG llama model"
    )
    parser.add_argument(
        "-b", "--build",
        help="Build the APS model",
        dest="build",
        default=False,
        action='store_true'
    )
    return parser.parse_args()

if __name__ == '__main__':
    ef = f'{Bcolors.ENDC}'
    bf = Bcolors.BOLD
    qf = f'{bf}{Bcolors.CYAN}'
    af = f'{bf}{Bcolors.MAGENTA}'
    qtf = Bcolors.WHITE
    atf = ''

    args = get_arguments()
    if args.build:
        result = create_model_with_system()
        print(result)
        quit()
        
    connection = initialize_database()
    stock_qs = [
        "How many rows are there?",
    ]
    history=[]
    for q in stock_qs:
        print(f'{qf}Asked:{ef} {qtf}{q}{ef}')
        response, history, _ = get_answer(connection, 'How many rows are there?', history=history)
        print(f'{af}Answered:{ef} {atf}{response}{ef}')
        print('')

    text ="x"
    while text:
        text = input(f'{qf}Ask:{ef} {qtf}')
        print(f'{ef}', end='')
        if not text: continue
        response, history, _ = get_answer(connection, text, history=history)
        print(f'{af}Answer:{ef} {atf}{response}{ef}')
        print('')

    print('Goodbye.')

    
