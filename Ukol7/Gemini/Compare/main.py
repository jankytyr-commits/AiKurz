import os
import requests
import json
import datetime
import wikipedia
import sympy
import math
import random
from tavily import TavilyClient
from dotenv import load_dotenv
import logging

# --- 0. Nastavení logování do souboru ---
LOGS_DIR = "Logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filepath = os.path.join(LOGS_DIR, f"chat_log_{timestamp}.log")

# Vytvoření Formatteru
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Nastavení Root Loggeru
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# 1. Handler pro soubor (všechny INFO detaily)
file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

# 2. Handler pro terminál (pouze WARNING a vyšší pro čistotu)
console_handler = logging.StreamHandler()
# Nastavení úrovně logování pro terminál
console_handler.setLevel(logging.WARNING) 
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

logging.info(f"Logování zahájeno do souboru: {log_filepath}")

# --- 1. Načtení API klíčů ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

# --- 2. Konfigurace API pro Gemini ---
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
HEADERS = {
    "Content-Type": "application/json",
    "x-goog-api-key": GOOGLE_API_KEY
}
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# --- 3. Definice nástrojů ---
def google_search(query: str):
    """
    Vyhledá informace na Google.
    """
    try:
        logging.info(f"Volání google_search, dotaz: '{query}'")
        print(f"-> Volám nástroj Google Search pro dotaz: '{query}'")
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_SEARCH_API_KEY,
            "cx": GOOGLE_SEARCH_ENGINE_ID,
            "q": query
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        search_results = response.json()
        if not search_results.get('items'):
            logging.warning("Google Search: Žádné výsledky nebyly nalezeny.")
            return "Žádné výsledky nebyly nalezeny."
        
        # Získání a zřetězení více URL a kontextu
        context_parts = []
        for i, item in enumerate(search_results['items'][:5]): 
            context_parts.append(f"ZDROJ {i+1}: {item['snippet']} (PLNÁ URL: {item['link']})")
        
        logging.info("Google Search úspěšný.")
        return "\n".join(context_parts)
    except requests.exceptions.RequestException as e:
        logging.error(f"Google Search SELHAL s chybou: {e}")
        return tavily_search(query)

def tavily_search(query: str):
    """
    Vyhledá informace na internetu pomocí Tavily API.
    """
    try:
        print(f"-> Volám nástroj Tavily Search pro dotaz: '{query}'")
        results = tavily_client.search(query=query, max_results=5)
        
        # Získání a zřetězení kontextu a URL ze všech 5 výsledků
        full_context_parts = []
        for i, result in enumerate(results['results'][:5]):
            full_context_parts.append(f"ZDROJ {i+1}: {result['content']} (PLNÁ URL: {result['url']})")
            
        final_context = "\n".join(full_context_parts)
        logging.info("Tavily Search úspěšný, vráceno %d výsledků." % len(results['results']))
        
        return final_context
    except Exception as e:
        logging.error(f"Tavily Search SELHAL s chybou: {e}")
        return f"Došlo k chybě při vyhledávání: {e}"

def calculator_tool(expression: str):
    """
    Vyhodnotí matematický výraz a vrátí výsledek.
    """
    if not any(op in expression for op in ['+', '-', '*', '/', '%', '**']) and not expression.replace('.', '', 1).isdigit():
        logging.error("Kalkulačka selhala: Vstup není platný matematický výraz.")
        return "Vstup není platný matematický výraz. Prosím, zadejte pouze matematickou rovnici."
        
    try:
        # Použití standardní Pythonovské syntaxe pro mocninu a math funkce
        # Oprava chyby: Zabezpečení proti operátorům, které nejsou v eval() bezpečné.
        result = eval(expression, {"math": math, "random": random})
        logging.info(f"Kalkulačka výpočet: {expression} = {result}")
        return str(result)
    except Exception as e:
        logging.error(f"Kalkulačka SELHALA s chybou: {e}")
        return f"Došlo k chybě při výpočtu: {e}"

def get_current_datetime():
    """
    Vrátí aktuální datum a čas.
    """
    now = datetime.datetime.now()
    result = now.strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Aktuální datum/čas: {result}")
    return result

def wikipedia_search(query: str):
    """
    Vyhledá informace na Wikipedii a vrátí shrnutí.
    """
    try:
        wikipedia.set_lang("cs")
        page = wikipedia.page(query, auto_suggest=True)
        result = page.summary
        logging.info(f"Wikipedia search successful for: {query}")
        return result
    except wikipedia.exceptions.DisambiguationError as e:
        logging.warning(f"Wikipedia DisambiguationError: {e.options}")
        return f"Bylo nalezeno více výsledků. Vyberte prosím jeden z těchto: {e.options}"
    except wikipedia.exceptions.PageError:
        logging.warning("Wikipedia PageError: Stránka nenalezena.")
        return "Stránka s tímto dotazem nebyla nalezena."
        
# --- 4. Funkce pro volání LLM a logiku ReAct s kontextem paměti ---
available_tools = {
    "google_search": google_search,
    "calculator_tool": calculator_tool,
    "get_current_datetime": get_current_datetime,
    "wikipedia_search": wikipedia_search,
}

def call_gemini_api_with_history(prompt: str, history: list, use_final_tool=False):
    """
    Volá Gemini API s daným promptem a historií konverzace.
    """
    tools_string = ', '.join(available_tools.keys())
    
    if use_final_tool:
        tools_string = ""

    full_prompt = f"""
    Jsi AI asistent, který odpovídá na dotazy uživatele.
    Máš přístup k následujícím nástrojům:
    {tools_string}

    Používej formát "Thought" (myšlenka) a "Action" (akce).

    Pokud potřebuješ použít nástroj, odešli zprávu ve formátu:
    Thought: Proč potřebuji nástroj
    Action: název_nástroje("parametr")
    
    Pokud nepotřebuješ nástroj a znáš odpověď, odešli pouze finální odpověď.

    PRAVIDLA PRO FINÁLNÍ ODPOVĚĎ:
    1. Odpověď musí být vždy formátována jako očíslovaný seznam.
    2. Každý bod seznamu musí být oddělen prázdným řádkem.
    3. Pokud odpověď obsahuje URL adresy, musí být uvedeny VŽDY v plném formátu (https://...).
    
    --- KONVERZACE ---
    {history}
    
    --- AKTUÁLNÍ DOTAZ ---
    {prompt}
    """
    
    # Odebereme z historie informace o mezikrocích
    simplified_history = []
    for turn in history:
        if turn['role'] == 'user' or turn['role'] == 'model':
            simplified_history.append(turn)

    contents = simplified_history + [{"role": "user", "parts": [{"text": full_prompt}]}]
    
    data = {
        "contents": contents
    }
    
    logging.info("Odesílám požadavek na Gemini API.")
    response = requests.post(GEMINI_URL, headers=HEADERS, json=data)
    response_json = response.json()
    
    if "error" in response_json:
        error_message = response_json['error']['message']
        logging.critical(f"Gemini API selhalo: {error_message}")
        raise Exception(f"Gemini API chyba: {error_message}")
        
    llm_output = response_json['candidates'][0]['content']['parts'][0]['text']
    logging.info("Gemini API vrátil odpověď.")
    return llm_output

def run_agent(user_input: str, history: list):
    """
    Spouští agenta, který rozhoduje o použití nástrojů s kontextem.
    """
    prompt = user_input
    
    for i in range(10):
        try:
            llm_output = call_gemini_api_with_history(prompt, history)
            
            logging.info(f"KROK {i+1} - LLM Výstup: {llm_output}")
            print(f"\n--- KROK {i+1} ---")
            print(f"**Myšlenka agenta:**\n{llm_output}")
            
            if "Action:" in llm_output:
                action_part = llm_output.split("Action:")[1].strip()
                if "(" in action_part and ")" in action_part:
                    action_name = action_part.split('("')[0]
                    action_arg = action_part.split('("')[1].strip('")')
                else:
                    action_name = action_part
                    action_arg = ""

                print(f"**Použitý nástroj:** {action_name}")
                logging.info(f"Použitý nástroj: {action_name}")
                
                if action_name in available_tools:
                    if action_arg:
                        tool_output = available_tools[action_name](action_arg)
                    else:
                        tool_output = available_tools[action_name]()
                else:
                    tool_output = f"Nástroj '{action_name}' nebyl nalezen."
                
                history.append({"role": "model", "parts": [{"text": llm_output}]})
                history.append({"role": "user", "parts": [{"text": f"Nástroj {action_name} vrátil: {tool_output}"}]})
                
                prompt = f"Nástroj {action_name} vrátil: {tool_output}"
                
            else:
                print("\n--- Konečná odpověď ---")
                return llm_output, history + [{"role": "model", "parts": [{"text": llm_output}]}]
                
        except Exception as e:
            logging.error(f"Chyba ve smyčce run_agent: {e}")
            return f"Došlo k chybě: {e}", history

    logging.warning("Agent vyčerpal limit 10 kroků.")
    print("Agent vyčerpal limit, spouštím FinalSolutionTool pro poslední pokus.")
    final_solution = call_gemini_api_with_history(user_input, history, use_final_tool=True)
    return f"\n--- Konečná odpověď ---\n{final_solution}", history

# --- 5. Hlavní smyčka pro uživatele ---
if __name__ == "__main__":
    print("Agent je připraven. Můžeš se ho ptát na cokoli, co vyžaduje vyhledávání.")
    conversation_history = []
    
    while True:
        query = input("\nTvůj dotaz: ")
        if query.lower() == "exit":
            break
        
        logging.info(f"Uživatel: {query}")
        conversation_history.append({"role": "user", "parts": [{"text": query}]})
        
        final_answer, conversation_history = run_agent(query, conversation_history)
        print(final_answer)