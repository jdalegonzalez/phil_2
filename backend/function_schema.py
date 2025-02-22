sql_query_schema = {
    "type": "function",
    "function": {
        "name": "sql_query",
        "description": "Execute a sqlite3 SQL query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The SQL string to execute"
                }
            },
            "required": ["query"]
        },
    },
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
            "required": [  "code"  ],
        },
    },
}

calculate_schema = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    },
}
