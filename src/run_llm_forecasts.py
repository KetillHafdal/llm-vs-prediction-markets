#This one id the one, 28/07 tekur listann af mörkuðum ogh flettiru þeim upp með 6 mismunandi llm og skilar excel skjali, tekur reyndar heila eilífð að virka, en tekur heila eilífð.
#Market 55,50,26 are all called
#historycal_response
#fara inn í forcasting tools og finna það sem ég er að nota, asknews serach og útskýra í texta hvað það gerir, Þetta er á github. minnir að þetta séu 67 frá the last 24 hours og 10 hiostorycal greinar sem ég fæ
#rename the last run
import os
import asyncio
import httpx
import json
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
#import sys
import tiktoken
import pandas as pd
from forecasting_tools import AskNewsSearcher
from asknews_client import AskNewsClient

load_dotenv()

class Question:
    def __init__(self, id: str, title: str, description: str):
        self.id = id
        self.title = title
        self.description = description

class OpenRouterLLM:
    def __init__(self, temperature: float = 0.3):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

        self.model = "openai/gpt-4o-mini"
        self.temperature = temperature
    async def chat(self, prompt: str, system_msg: Optional[str] = "You are a helpful forecasting assistant. Respond only with a number between 0 and 1.") -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://chat.openai.com",
        }

        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ]
        }

        for attempt in range(3):  # Retry up to 3 times
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"].strip()
            except httpx.ReadTimeout:
                print(f"⚠️ Timeout on attempt {attempt + 1} for model {self.model}. Retrying...")
                await asyncio.sleep(2)
            except Exception as e:
                print(f"❌ Unexpected error for model {self.model}: {e}")
                break

        return ""    
'''
    async def chat(self, prompt: str, system_msg: Optional[str] = "You are a helpful forecasting assistant. Respond only with a number between 0 and 1.") -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://chat.openai.com",
        }

        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ]
        }

        async with httpx.AsyncClient() as client:
            response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
            #print("📨 LLM HTTP status:", response.status_code)
            #print("📨 LLM raw response:", response.text[:500])

            try:
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"❌ Failed to parse LLM response JSON: {e}")
                return ""
'''
def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

class MyCustomForecastBot:
    def __init__(self, model_names: List[str], use_summarizer: bool = True):
        self.model_names = model_names
        self.use_summarizer = use_summarizer
        self.results = []

    def load_questions_from_json(self, filepath: str) -> List[Question]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Question(q["id"], q["question"], q["description"]) for q in data]

    def get_custom_questions(self) -> List[Question]:
        return self.load_questions_from_json("open_markets_wip.json") #where I get the questions from.
        #return self.load_questions_from_json("open_markets_last_10.json") #where I get the questions from.

    async def run_research(self, question_text: str) -> str:
        try:
            searcher = AskNewsSearcher()
            research = await searcher.get_formatted_news_async(question_text)
            print("🔍 AskNewsAPI Research Output:\n")
            return research
        except Exception as e:
            print(f"❌ Research failed: {e}")
            return ""

    async def predict(self, question: Question) -> List[dict]:
        results = []

        research = await self.run_research(question.title)
        #print("📚 Research:\n", research)

        if not research.strip():
            print(f"⚠️ Skipping {question.id} — no research available.")
            return results

        token_count = count_tokens(research, model_name="gpt-3.5-turbo")
        print(f"📊 Token count for research: {token_count}")

        summary = research
        if self.use_summarizer:
            summarizer_llm = OpenRouterLLM()
            summarizer_llm.model = self.model_names[0]

            summary_prompt = (
                f"Extract and synthesize key insights from the following research text. "
                f"Focus on trends, arguments, and signals relevant to forecasting. "
                f"Do not assign any probabilities or predictions.\n\n"
                f"Research:\n{research}"
            )

            summary = await summarizer_llm.chat(
                summary_prompt,
                system_msg="You are a summarization assistant. Your task is to condense and extract meaningful insights from news data for a forecasting assistant. Do not provide a numeric probability or any prediction."
            )
            print("📝 Summary (shared by all models):\n", summary)

        row = {
            "id": question.id,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "category": "N/A",
            "end_date": "N/A",
            "question": question.title,
            "description": question.description,
            "research_tokens": token_count
        }

        for model_name in self.model_names:
            for style in ["zero_shot", "chain_of_thought"]:
                print(f"🔁 Using model: {model_name} with style: {style}")
                llm = OpenRouterLLM()
                llm.model = model_name

                if style == "zero_shot":
                    forecast_prompt = (
                        f"You're a forecasting assistant.\n\n"
                        f"Question: {question.title}\n"
                        f"Description: {question.description}\n\n"
                        f"Research Summary:\n{summary}\n\n"
                        f"Based on this information, estimate the probability this event will occur.\n"
                        f"Respond only with a number between 0 and 1, and nothing but a number, no text just this number between 0 and 1!."
                    )
                else:
                    forecast_prompt = (
                        f"You're a careful and analytical forecasting assistant.\n\n"
                        f"Question: {question.title}\n"
                        f"Description: {question.description}\n\n"
                        f"Research Summary:\n{summary}\n\n"
                        f"Think step by step. Consider all relevant information before giving a final answer.\n"
                        f"At the end, estimate the probability this event will occur. Respond only with a number between 0 and 1, It is very important that you dont include any text in you final answear only the number."
                    )

                try:
                    response = await llm.chat(forecast_prompt)
                    prediction = float(response)
                except Exception as e:
                    print(f"❌ Prediction failed for {question.id} with {model_name} ({style}): {e}")
                    prediction = None

                row[f"{model_name}_{style}"] = prediction

        return row
    '''
    async def run(self):
        all_rows = []

        for question in self.get_custom_questions():
            predictions = await self.predict(question)

            if isinstance(predictions, list):
                all_rows.extend(predictions)
            else:
                print(f"⚠️ Unexpected result from predict() for {question.id}: {type(predictions)}")

        df = pd.DataFrame(all_rows)
        timestamp_str = datetime.utcnow().strftime("%Y-%m-%d")
        df.to_excel(f"forecast_results_{timestamp_str}.xlsx", index=False)
        print(f"✅ Results saved to forecast_results_{timestamp_str}.xlsx")
    '''
    async def run(self):
        all_rows = []
        for question in self.get_custom_questions():
            result_row = await self.predict(question)
            all_rows.append(result_row)
            

        df = pd.DataFrame(all_rows)
        timestamp_str = datetime.utcnow().strftime("%Y-%m-%d")
        df.to_excel(f"forecast_results_{timestamp_str}.xlsx", index=False)
        print(f"✅ Results saved to forecast_results_{timestamp_str}.xlsx")

    

if __name__ == "__main__":
    models_to_test = [
        "openai/gpt-4o-mini",
        "openai/gpt-3.5-turbo-0613",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it:free",
        #"google/gemma-2-9b-it:free",Dont use
        #"deepseek/deepseek-r1:free" Takes forever
        "deepseek/deepseek-chat-v3-0324:free",
        "deepseek/deepseek-r1:free"
    ]

    use_summarizer = True

    asyncio.run(MyCustomForecastBot(models_to_test, use_summarizer).run())









'''
import os
import asyncio
import httpx
import csv
import json
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
import sys
from forecasting_tools import AskNewsSearcher
from asknews_client import AskNewsClient
import tiktoken
import re

load_dotenv()

class Question:
    def __init__(self, id: str, title: str, description: str):
        self.id = id
        self.title = title
        self.description = description

class OpenRouterLLM:
    def __init__(self, temperature: float = 0.3):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

        self.model = "openai/gpt-4o-mini"
        self.temperature = temperature

    async def chat(self, prompt: str, system_msg: Optional[str] = "You are a helpful forecasting assistant. Respond only with a number between 0 and 1.") -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://chat.openai.com",
        }

        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ]
        }

        async with httpx.AsyncClient() as client:
            response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
            #print("📨 LLM HTTP status:", response.status_code)
            #print("📨 LLM raw response:", response.text[:500])

            try:
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"❌ Failed to parse LLM response JSON: {e}")
                return ""

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def extract_float_from_text(text: str) -> Optional[float]:
    match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    return float(match.group(1)) if match else None

class MyCustomForecastBot:
    def __init__(self, model_names: List[str], use_summarizer: bool = True):
        self.model_names = model_names
        self.use_summarizer = use_summarizer
        self.results = []

    def load_questions_from_json(self, filepath: str) -> List[Question]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Question(q["id"], q["question"], q["description"]) for q in data]

    def get_custom_questions(self) -> List[Question]:
        return self.load_questions_from_json("open_markets_filtered.json")

    async def run_research(self, question_text: str) -> str:
        try:
            searcher = AskNewsSearcher()
            research = await searcher.get_formatted_news_async(question_text)
            print("🔍 AskNewsAPI Research Output:\n")
            return research
        except Exception as e:
            print(f"❌ Research failed: {e}")
            return ""

    async def predict(self, question: Question) -> List[dict]:
        results = []

        research = await self.run_research(question.title)
        #print("📚 Research:\n", research)

        if not research.strip():
            print(f"⚠️ Skipping {question.id} — no research available.")
            return results

        token_count = count_tokens(research, model_name="gpt-3.5-turbo")
        print(f"📊 Token count for research: {token_count}")

        summary = research
        if self.use_summarizer:
            summarizer_llm = OpenRouterLLM()
            summarizer_llm.model = self.model_names[0]

            summary_prompt = (
                f"Extract and synthesize key insights from the following research text. "
                f"Focus on trends, arguments, and signals relevant to forecasting. "
                f"Do not assign any probabilities or predictions.\n\n"
                f"Research:\n{research}"
            )

            summary = await summarizer_llm.chat(
                summary_prompt,
                system_msg="You are a summarization assistant. Your task is to condense and extract meaningful insights from news data for a forecasting assistant. Do not provide a numeric probability or any prediction."
            )
            print("📝 Summary (shared by all models):\n", summary)

        for model_name in self.model_names:
            for style in ["zero_shot", "chain_of_thought"]:
                print(f"🔁 Using model: {model_name} with style: {style}")
                llm = OpenRouterLLM()
                llm.model = model_name

                if style == "zero_shot":
                    forecast_prompt = (
                        f"You're a forecasting assistant.\n\n"
                        f"Question: {question.title}\n"
                        f"Description: {question.description}\n\n"
                        f"Research Summary:\n{summary}\n\n"
                        f"Based on this information, estimate the probability this event will occur.\n"
                        f"Respond only with a number between 0 and 1, make sure not to include anything else than a number between 0 and 1 in your response."
                    )
                else:  # chain_of_thought
                    forecast_prompt = (
                        f"You're a careful and analytical forecasting assistant.\n\n"
                        f"Question: {question.title}\n"
                        f"Description: {question.description}\n\n"
                        f"Research Summary:\n{summary}\n\n"
                        f"Think step by step. Consider all relevant information before giving a final answer.\n"
                        f"At the end, estimate the probability this event will occur. Respond only with a number between 0 and 1."
                    )

                try:
                    
                    response = await llm.chat(forecast_prompt)
                    prediction = extract_float_from_text(response)
                    if prediction is None:
                        print(f"❌ Could not extract float from response for {question.id} with {model_name} ({style})")
                except Exception as e:
                    print(f"❌ Prediction failed for {question.id} with {model_name} ({style}): {e}")
                    prediction = None

                results.append({
                    "id": question.id,
                    "title": question.title,
                    "model": model_name,
                    "style": style,
                    "prediction": prediction,
                    "timestamp": datetime.utcnow().isoformat(),
                    "research_tokens": token_count
                })

        return results

    async def run(self):
        for question in self.get_custom_questions():
            predictions = await self.predict(question)
            for result in predictions:
                print(f"[{result['model']} - {result['style']}] {result['title']} → {result['prediction']}")

if __name__ == "__main__":
    models_to_test = [
        "openai/gpt-4o-mini",
        "openai/gpt-3.5-turbo-0613",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it:free",
        #"google/gemma-2-9b-it:free",
        #"deepseek/deepseek-r1:free" Takes forever
        "deepseek/deepseek-chat-v3-0324:free",
        "deepseek/deepseek-r1:free"

        
        
    ]

    use_summarizer = True

    asyncio.run(MyCustomForecastBot(models_to_test, use_summarizer).run())
'''
