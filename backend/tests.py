import re
import unittest
from agent import run_generate_graph, get_connection

class TestGraph(unittest.TestCase):
    def test_from_code(self):
        conn = get_connection()
        #graph_code = "import pandas as pd\nresult = sql_query('SELECT School_Name, AVG(Total_Score) AS Average_Total_Score FROM eval GROUP BY School_Name')\ndata = pd.DataFrame(result)\ndata.plot(kind='bar', x='School_Name', y='Average_Total_Score', figsize=(10,6))"
        #self.assertNotEqual(run_generate_graph(conn, graph_code),"Something went wrong.  Please try again.")
        #graph_code = "import pandas as pd\nfrom matplotlib import pyplot as plt\ndata = sql_query(query='SELECT School_Name, Total_Score FROM eval')\nplt.bar(data['School_Name'], data['Total_Score'])\nplt.xlabel('School Name')\nplt.ylabel('Total Score')\nplt.title('Total Scores by School')"
        #self.assertNotEqual(run_generate_graph(conn, graph_code),"Something went wrong.  Please try again.")
        graph_code = "import matplotlib.pyplot as plt\nfrom pandas import DataFrame\nresults = sql_query(query='SELECT School_Name, AVG(Total_Score) FROM eval GROUP BY School_Name')\nDataFrame(results).plot(kind='bar', x='School_Name', y='AVG(Total_Score)').show()\n"
        self.assertNotEqual(run_generate_graph(conn, graph_code),"Something went wrong.  Please try again.")


if __name__ == "__main__": unittest.main()