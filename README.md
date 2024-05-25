# PORTFOLIO_py_openai_api

Exploring OpenAI API

## Installing

~~~
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt 
~~~

## Running

Create a file named '.env' setting the you OPENAI_API_KEY license. i.e.:

~~~bash
OPENAI_API_KEY=sk-........................
~~~
Then:
~~~
export PYTHONPATH=./:$PYTHONPATH
python tests/test_openai_api.py
~~~
