import re
import unittest
import json
from agent import run_generate_graph, get_connection

qwen_re = re.compile("^\\<tool_call\\>\n(.*)\n\\<\\/tool_call\\>")
llama_re = re.compile("^\\<function=(.*)\\<\\/function\\>")
llama_re = re.compile("function")
class Message:
    def __init__(self, content):
        self.content = content

def parse_tool_calls(message):

    reg = llama_re
    print("*******")
    print(message.content)    
    #OK, qwen tool calls look <tool_call>\n{json stuff}\n</tool_call>
    # This code won't work if there is more than one tool call in the content.
    # I've never seen a string like that, so I don't know exactly how to 
    # change the regex.
    print("=======")
    print(reg.match(message.content))
    print("----------")
    c2 = match.groups(0)[0] if (match := reg.match(message.content)) else ""
    print(c2)

    if not c2: return message
    
    try:
        call = json.loads(c2)
        print(call)
        # In OpenAI land, arguments are a string, not a json obj - WHICH SUCKS
        call['arguments'] = json.dumps(call['arguments'])
        tool_call = {'id': '123', 'function': call, 'type': 'function'}
        print([tool_call])
    
    except Exception:
        # OK, it's not json, so....
        print("ERROR")

    
    return message


class TestGraph(unittest.TestCase):

    def test_from_code(self):
        conn = get_connection()
        #graph_code = "import pandas as pd\nresult = sql_query('SELECT School_Name, AVG(Total_Score) AS Average_Total_Score FROM eval GROUP BY School_Name')\ndata = pd.DataFrame(result)\ndata.plot(kind='bar', x='School_Name', y='Average_Total_Score', figsize=(10,6))"
        #self.assertNotEqual(run_generate_graph(conn, graph_code),"Something went wrong.  Please try again.")
        #graph_code = "import pandas as pd\nfrom matplotlib import pyplot as plt\ndata = sql_query(query='SELECT School_Name, Total_Score FROM eval')\nplt.bar(data['School_Name'], data['Total_Score'])\nplt.xlabel('School Name')\nplt.ylabel('Total Score')\nplt.title('Total Scores by School')"
        #self.assertNotEqual(run_generate_graph(conn, graph_code),"Something went wrong.  Please try again.")
        graph_code = "import matplotlib.pyplot as plt\nfrom pandas import DataFrame\nresults = sql_query(query='SELECT School_Name, AVG(Total_Score) FROM eval GROUP BY School_Name')\nDataFrame(results).plot(kind='bar', x='School_Name', y='AVG(Total_Score)').show()\n"
        self.assertNotEqual(run_generate_graph(conn, graph_code),"Something went wrong.  Please try again.")

    def test_parsing_llama(self):
        message = Message('### Number of Rows\nTo get the number of rows in the system, we need to run a SQL query. \n<function=sql_query{"query": "SELECT COUNT(*) FROM eval"}</function> \n\n**Here are some suggestions:**\n[Get the count of schools.](#)\n[Get the count of Grade Levels.](#)')
        result = parse_tool_calls(message)
        print(result)

if __name__ == "__main__": unittest.main()