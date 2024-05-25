# PORTFOLIO_py_openai_api

An OpenAI API interface that implements the methods:

**set_openai_key()**

Set openai key using environment variable OPENAI_KEY inside .env file

**get_completion(prompt, model, temperature)**

Get models response using OpenAI API

Parameters:
- prompt: prompt to send to OpenAI
- model: OpenAI model to use
- temperature: degree of randomness/creativity of the model's output (0= no randomness, 1=super-random)

Returns
- message: is the response message or the exception code
- finish_reason: is the reason why the message was returned



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
