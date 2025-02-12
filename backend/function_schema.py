
sql_query_schema = {
    "type": "function",
    "function": {
        "name": "sql_query",
        "description": "Run a SQL SELECT query on a SQLite database and return the results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL compatible with SQLite as a string"
                }
            },
            "required": [
                "query"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}

generate_graph_schema = {
    "type": "function",
    "function": {
        "name": "generate_graph",
        "description": "Takes python code that calls matplotlib.pyplot to generate a chart or graph, runs the python code in interpreter and returns an SVG version of the graph",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "python code that uses matplotlib.pyplot to generate a graph"
                }
            },
            "required": [
                "code"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}
