import openai
from flask import Flask, redirect, render_template, request, url_for
import base64
import sys
import traceback
from lib import DictArrayManager
import importlib
import os
import csv
from dotenv import load_dotenv
from prompts import system_prompt
from utils import get_context, count_tokens, past_reflect, CHAT_API_PARAMS
import pandas as pd
import numpy as np

print(pd. __version__)
# load_dotenv()
#OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

app = Flask(__name__)
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY



document_chunks = pd.read_csv('data/chunked_texts.csv')
embeddings_pkl = pd.read_pickle('data/embeddings.pkl')

# openai.api_key = OPENAI_API_KEY

# Set the number of past tokens to send with the current query
# rounds to the nearest whole message
history_max_tokens = 14000

global messages
messages = DictArrayManager()
messages.add("system", system_prompt)

full_context = []

@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        if 'reset' in request.form:
            reset()
        elif len(request.form["query"]) > 0:
            return process_request(request,full_context)
    return render_template("index.html", anchor='query', messages=messages.array, tokens=messages.tokens, n=len(messages.array))

def reset():
    # Reload openai module
    importlib.reload(openai)
    openai.api_key = OPENAI_API_KEY
    messages.clear()
    full_context = []
    messages.add("system", system_prompt)

def process_request(request,full_context):
    global messages
    model = request.form["model"]
    user_message = request.form["query"]+'skew deviation'
    messages.add("user", user_message)
    context = get_context(user_message,embeddings_pkl,document_chunks)
    try:
        context_messages = messages     
        if len(full_context) > 0:
            full_context = list(set(full_context.extend(context)))
            context = ''
            for chunk in full_context:
                context = context+chunk
        else:
            full_context = context
            temp = ''
            for chunk in context:
                temp = temp+chunk
            context = temp
        context = '```'+context+'```'
        context_messages.add("system", context)
        response = openai.ChatCompletion.create(model=model, messages=context_messages.array, **CHAT_API_PARAMS)
        response_message = response.choices[0].message["content"]
        messages.add("assistant", response_message)
        tokens = 0
        history_max_tokens = 14000
        for message in full_context:
            tokens += count_tokens(context)
            tokens += count_tokens(message)
            if tokens > history_max_tokens:
                reflection = past_reflect(context_messages.array)
                full_context = reflection
                break
        tokens = 0

        
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        stack = ''.join('!! ' + line for line in lines)
        messages.add("system", "ERROR:\n" + stack)
        print("handled exception\n", stack)
    return redirect(url_for('index', _anchor='query'))

@app.route('/log')
def view_log():
    log_lines = []
    with open('log.csv', newline='') as log_file:
        csv_reader = csv.DictReader(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csv_reader:
            for i in row:
                if is_base64(row[i]):
                    row[i] = base64.b64decode(row[i]).decode("utf-8")
            log_lines.append(row)
    return render_template('log.html', log_lines=log_lines)

def is_base64(s):
    try:
        decoded_string = base64.b64decode(s).decode("utf-8")
        if messages.encode(decoded_string) == s:
            return True
        else:
            return False
    except:
        return False

reset()