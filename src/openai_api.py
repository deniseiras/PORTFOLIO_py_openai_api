"""
Open AI API

:author: Denis Eiras

Functions:
    - get_completion: get models response using OpenAI API
    - set_openai_key: set openai key using environment variable OPENAI_KEY inside .env file
"""

import openai
import dotenv
import os
import tiktoken


def set_openai_key():
    """
        set openai key using environment variable OPENAI_KEY inside.env file
    """
    if openai.api_key is None:
        print('setting openai key')
        del os.environ['OPENAI_API_KEY']
        # load the OPENAI_API_KEY in .env file
        dotenv.load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')


def get_completion(prompt_user, prompt_system=None, model="gpt-4o-mini", temperature=0):
    """
        get completion using OpenAI

        :param prompt: prompt to send to OpenAI
        :param model: OpenAI model to use
        :param temperature: degree of randomness/creativity of the model's output (0= no randomness, 1=super-random)
        :return: message, finish_reason
            - message is the response message or the exception code
            - finish_reason is the reason why the message was returned
        
        Every response will include a finish_reason. The possible values for finish_reason are:
        - stop: API returned complete message, or a message terminated by one of the stop sequences provided via the stop parameter
        - length: Incomplete model output due to max_tokens parameter or token limit
        - function_call: The model decided to call a function
        - content_filter: Omitted content due to a flag from our content filters
        - null: API response still in progress or incomplete
        - 'openai_exception' will be raised when there is an OpenAI exception

        Whether your API call works at all, as total tokens must be below the model’s maximum limit:
        - 4096 tokens for   gpt-3.5-turbo-instruct - US$ 1,50/1M input tokens, US$ 2,00/1M output tokens 
        - 16k tokens for    gpt-3.5-turbo-0125 - US$ 0,50 / 1M input tokens
        - 128k tokens for   gpt-4 - US$ 5,00 / 1M input tokens , US$ 15,00 / 1M output tokens
        - 128k tokens for   gpt-4o-mini - Input: $0.15 | Output: $0.60* https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/
        
        * gpt-3.5-turbo actually points to gpt-3.5-turbo-instruct . For some reason in this API gpt-3.5-turbo doesn't uses the latest version (gpt-3.5-turbo-0125)
        
        
    """
        
    set_openai_key()
    
    mssgs = [{"role": "user", "content": prompt_user}]
    if prompt_system:
        mssgs.append({"role": "system", "content": prompt_system})
    try:
        client = openai.OpenAI()
        
        response = client.chat.completions.create(
            model=model,
            # response_format={"type": "json_object"},
            messages=mssgs,
            temperature=temperature,
        )
        
        choice = response.choices[0]
        ret_message = choice.message.content
        ret_fin_reason = choice.finish_reason
        
    except Exception as e:
        ret_message = e.code
        ret_fin_reason = 'openai_exception'

    # TODO extract the model name for this. gpt-3.5-turbo does not work here
    tokenizer = tiktoken.get_encoding('cl100k_base')
    if prompt_system:
        tokens_sys = tokenizer.encode(prompt_system)
    else:
        tokens_sys = []
    tokens_user = tokenizer.encode(prompt_user)
    tokens_response = tokenizer.encode(ret_message)
    
    print(f'\n\nPrompt System: {len(tokens_sys)} tokens.\n ===> {prompt_system}')
    print(f'Prompt User:  {len(tokens_user)} tokens.\n ===> {prompt_user}')
    print(f'Tokens Response: {len(tokens_response)} tokens.\n')
    print(f'Total tokens = {len(tokens_sys)+len(tokens_user)}')
    print(f'\n==============\nResponse: {ret_message}')
    print(f'\nFinish reason: {ret_fin_reason}')
    
    return ret_message, ret_fin_reason

