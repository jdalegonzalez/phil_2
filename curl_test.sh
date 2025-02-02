#!/usr/bin/bash

if [ "$1" == "" ]; then
    echo "Usage: $0 <question text>"
    exit 1
fi
query=$(echo -n "$*" | jq -sRr @uri)
curl --cookie ./cookies.txt --cookie-jar ./cookies.txt http://127.0.0.1:8000/?q=$query
