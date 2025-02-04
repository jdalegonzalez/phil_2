import pandas
import matplotlib
import base64

import io
import re
from datetime import datetime
import sqlite3
import argparse
import logging

from ollama import Client, chat, ChatResponse

DEFAULT_MODEL = 'aps'
DEFAULT_BASE  = 'llama3.1:8b'

system_prompt = """
You are an expert SQL analyst and python data-scientist. When appropriate, generate SQL queries based on the user question and the database schema.
When you generate a query, use the 'sql_query' function to execute the query on the database and get the results. Then, use the results to answer the user's question. 
You should only use SQL supported by sqlite3.  DO NOT try to fetch graphs or tables from the internet.

* There is only one table so don't bother mentioning its name. Instead of talking about a database, use the word system.
* When you're generating table headings, replace underscores ('_') with spaces. 

* ALWAYS FORMAT YOUR RESPONSE IN MARKDOWN

* To create a graph, chart or plot use ONLY python, sql_query, pandas, and matplotlib. Do NOT use sqlalchemy.  You do not need to call plt.show()

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
  * Do present the data in the form of a table unless asked otherwise.
  * When asked for Total Score DO provide an average for the grouping of data
  * Do provide the average score when asked for scores e.g. If asked to summarize the data by School Name and there are 5 entries for Wesley School then you would present the average of the five. 
  * If you calculate an average scores just provide the score, do not display the calculation behind the average unless you are asked

SUGGESTIONS:
  * Do present at least 2 helpful suggestions after every output for showing more analysis. 
    Provide the suggests as hyperlinks in the format: [This is a suggestion.](#). Before the suggestions, include the text: '**Here are some suggestions:**'.  

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

def run_generate_graph(connection: sqlite3.Connection, code: str) -> str:
    """
    Executes a block of python code that is expected to use matplotlib to generate a graph
    and then returns an svg tag that can be displayed to the user.

    Args:
        code: The block of python code that can generate an image

    Returns:
        str: A markdown, embeddable SVG
    """
    
    def sql_query(query:str) -> dict:
        return run_sql_query(connection, query)
    

    loc = dict(locals())
    exec(code, globals(), loc)
    if (plt := loc.get('plt')):
        # Save the plot to a BytesIO object in SVG format
        svg_buffer = io.BytesIO()
        plt.savefig(svg_buffer, format='svg')
        # Get the string representation of the SVG
        svg_string = svg_buffer.getvalue().decode()        
        res = f'![svg image](data:image/svg+xml;base64,{base64.b64encode(svg_string.encode('utf-8'))})'
        print(res)
        return res
    
    return "![svg image](data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20width%3D%22100%22%20height%3D%22100%22%20viewBox%3D%220%200%20100%20100%22%3E%3Ccircle%20cx%3D%2250%22%20cy%3D%2250%22%20r%3D%2240%22%20stroke%3D%22black%22%20stroke-width%3D%223%22%20fill%3D%22red%22%20/%3E%3C/svg%3E%0A)"

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

# messages = [{'role': 'system', 'content': 'What is three plus one?'}]

def convert_python_if_necessary(connection: sqlite3.Connection, message: str):
    """
    Looks for strings that intend to produce charts and redirects
    to charting code here.  I REALLY wanted to be fancy and get the
    agent to call my code automatically but it is refusing to do it.
    SO...

    Args:
        connection: The active database connection
        message: The message returned from the LLM
    
    Returns:
        str: Either the message unchanged or the code run and
             converted into svg
    """
    m = message.strip()
    print("Starts ok", m.startswith('```python'))
    print("Ends OK", m.endswith('```'))
    print(m[-4:])
    print("Imports", "import matplotlib.pyplot" in m)
    if not m.startswith('```python') or not m.endswith('```'):
        return message
    
    #OK, it's at least python.
    if "import matplotlib.pyplot" not in m: return message

    #OK, it's attempting to do something with plotting.
    # Try to find the result
    return run_generate_graph(connection, m)

def get_answer(
    connection:sqlite3.Connection, 
    question:str, history: list[dict] = [], 
    model: str = DEFAULT_MODEL, final_model: str = None) -> tuple[str, list[dict], ChatResponse]:

    if final_model is None: final_model = model

    def sql_query(query: str):
        """
        Runs an SQL query and returns the result as a dict

        Args:
            query: The query to be run

        Returns:
            dict: The results object
        """
        return run_sql_query(connection, query)
   
    def generate_graph(code: str):
        """
        Executes a block of python code to generate charts and graphs.  The 
        code is expected to use matplotlib and pandas, along with the sql_query
        function.

        Args:
            code: The python code to generate the graph or chart

        Returns:
            str: an embeddale svg.
        """
        return run_generate_graph(connection, code)
   
    history.append({'role': 'user', 'content': question})
    available_functions = {
        'sql_query': sql_query,
        'generate_graph': generate_graph,
    }
    response = chat(
        model=model,
        messages=history,
        tools=list(available_functions.values()),
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
            else:
                logging.warning(f"Function '{tool.function.name}' not found")

        history.append(response.message)
        history.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})
        response = chat(final_model, messages=history)

    converted = convert_python_if_necessary(history, response.message.content)
    print(converted)
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

    
