"""
Open AI API

:author: Denis Eiras

Functions:
    - get_completion: get completion using OpenAI
    - set_openai_key: set openai key using environment variable OPENAI_KEY inside .env file
"""

import openai
import dotenv
import os


def set_openai_key():
    if openai.api_key is None:
        print('setting openai key')
        del os.environ['OPENAI_API_KEY']
        # load the OPENAI_API_KEY in .env file
        dotenv.load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')


def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
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
        - 4097 tokens for gpt-3.5-turbo
        - 128k tokens for gpt-4
    """
        
    set_openai_key()
    
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            # TODO # response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        choice = response.choices[0]
        ret_message = choice.message.content
        ret_fin_reason = choice.finish_reason
    except Exception as e:
        ret_message = e.code
        ret_fin_reason = 'openai_exception'
        
    print(f'\nPrompt: {prompt}')
    print(f'Response: {ret_message}')
    print(f'Finish reason: {ret_fin_reason}')
    return ret_message, ret_fin_reason

