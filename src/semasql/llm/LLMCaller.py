import os
import time
import json
import asyncio
import tiktoken


from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class LLMCaller:
    def __init__(self, model, max_concurrency = 10):
        try:
            with open('src/semasql/conf/configure.json', 'r') as file:
                conf = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError("Configuration file './src/semasql/conf/configure.json' not found")
        
        self.model = model
        self.temperature = float(conf['temperature'])
        self.stream = str(conf['stream']).lower() == 'true'
        self.max_tries = int(conf['max_tries'])

        self.total_tokens_used = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.semaphore = asyncio.Semaphore(max_concurrency)

        api_key = os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('OPENAI_BASE_URL')

        if self.model == 'qwen':
            api_key = os.getenv('QWEN_API_KEY')
            self.model = "qwen3-235b-a22b-instruct-2507"
        elif self.model == 'gemini':
            self.model = "gemini-3-pro-preview"
        elif self.model == 'claude':
            self.model = "claude-sonnet-4-5-20250929"
        elif self.model == 'gpt':
            self.model = "gpt-5-2025-08-07"
        else:
            raise ValueError(f"Model {self.model} not recognized. Available models: 'qwen', 'gemini', 'claude', 'gpt'")

        
        self.client = OpenAI(
            api_key = api_key, 
            base_url = base_url,
            timeout = 30 * 60 # 30 minutes
        )

    def _get_encoding(self):
        """
        Get appropriate tiktoken encoding for token counting.
        
        Returns:
            tiktoken encoding object, or None if model doesn't support tiktoken
        """
        if self.model.startswith('gpt'):
            try:
                return tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fallback to cl100k_base for newer GPT models
                return tiktoken.get_encoding("cl100k_base")
        elif self.model.startswith('claude'):
            # Claude uses a similar tokenizer to GPT-4, use cl100k_base as approximation
            return tiktoken.get_encoding("cl100k_base")
        else:
            # Qwen, Gemini, and other models don't have tiktoken support
            # Use cl100k_base as a rough approximation
            return tiktoken.get_encoding("cl100k_base")

    def call(self, query, temperature = 0):
        tries = 0
        while tries < self.max_tries:
            try:
                if not self.stream:
                    if self.model.startswith('claude'):
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=query,
                            max_tokens= 50000,
                            temperature = self.temperature
                        )
                    else:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=query,
                            temperature = self.temperature
                        )
                    if response.usage:
                        self.total_tokens_used += response.usage.total_tokens
                        self.input_tokens += response.usage.prompt_tokens
                        self.output_tokens += response.usage.completion_tokens
                    return response.choices[0].message.content
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=query,
                        temperature=self.temperature,
                        stream=True
                    )

                    full_response = ""
                    for chunk in response:
                        if hasattr(chunk, 'choices') and chunk.choices is not None and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content:
                                full_response += delta.content

                    encoding = self._get_encoding()
                    input_token = len(encoding.encode("".join([f"{m['role']}: {m['content']}\n" for m in query])))
                    output_token = len(encoding.encode(full_response))
                    self.total_tokens_used += input_token + output_token
                    self.input_tokens += input_token
                    self.output_tokens += output_token

                    return full_response
            except Exception as e:
                tries += 1
                print(f"Error occurred while processing query: {e}. Retrying ({tries}/{self.max_tries})...")
                time.sleep(1)


    async def async_call(self, query):
        async with self.semaphore:
            tries = 0
            while tries < self.max_tries:
                try:
                    if not self.stream:
                        if self.model.startswith('claude'):
                            response = await asyncio.to_thread(lambda: self.client.chat.completions.create(model=self.model, messages=query, temperature=self.temperature, max_tokens= 50000))
                        else:
                            response = await asyncio.to_thread(lambda: self.client.chat.completions.create(model=self.model, messages=query, temperature=self.temperature))
                        
                        
                        if response.usage:
                            self.total_tokens_used += response.usage.total_tokens
                            self.input_tokens += response.usage.prompt_tokens
                            self.output_tokens += response.usage.completion_tokens
                        return response.choices[0].message.content
                    else:
                        response = await asyncio.to_thread(lambda: self.client.chat.completions.create(model=self.model, messages=query, temperature=self.temperature, stream=True))
                        full_response = ""
                        for chunk in response:
                            if hasattr(chunk, 'choices') and chunk.choices is not None and len(chunk.choices) > 0:
                                delta = chunk.choices[0].delta
                                if hasattr(delta, 'content') and delta.content:
                                    full_response += delta.content

                        encoding = self._get_encoding()
                        input_token = len(encoding.encode("".join([f"{m['role']}: {m['content']}\n" for m in query])))
                        output_token = len(encoding.encode(full_response))
                        self.total_tokens_used += input_token + output_token
                        self.input_tokens += input_token
                        self.output_tokens += output_token

                        return full_response
                except Exception as e:
                    tries += 1
                    print(f"Error occurred while processing query: {e}. Retrying ({tries}/{self.max_tries})...")
                    await asyncio.sleep(1) 


    async def call_batch_async(self, queries):
        async def delayed_call(query):
            await asyncio.sleep(5)
            return await self.async_call(query)
        tasks = [delayed_call(query) for query in queries]
        results = await asyncio.gather(*tasks)
        return results


    def get_total_tokens_used(self):
        return self.total_tokens_used, self.input_tokens, self.output_tokens