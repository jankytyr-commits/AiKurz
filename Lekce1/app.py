import os
import sys
import json
import codecs
import http.client # Import pro monkey-patching
import traceback # Import pro detailní výpis chyb
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Optional

# --- Oprava pro Unicode, LangSmith a Pylance ---

# --- ZÁSADNÍ OPRAVA pro UnicodeEncodeError ("monkey-patch") ---
# Tato sekce programově opravuje chybu v nízkoúrovňové knihovně http.client,
# která způsobuje pád při odesílání dat s diakritikou.
def patch_http_client_putheader():
    # Uložíme si původní, problematickou metodu
    original_putheader = http.client.HTTPConnection.putheader

    # Vytvoříme novou, opravenou verzi metody
    def new_putheader(self, header, *values):
        # Pokusíme se zakódovat všechny hodnoty v hlavičce do UTF-8
        try:
            values = [v.encode('utf-8') if isinstance(v, str) else v for v in values]
        except UnicodeEncodeError:
            # Pokud selže, vrátíme se k původnímu chování, ale chyba pravděpodobně nastane znovu
            pass
        
        # Zavoláme původní metodu s již bezpečně zakódovanými daty
        return original_putheader(self, header, *values)

    # Nahradíme původní metodu naší novou verzí
    http.client.HTTPConnection.putheader = new_putheader
    print("INFO: HTTP client has been patched to handle UTF-8 correctly.")

# Zavoláme naši opravnou funkci
patch_http_client_putheader()


# 1. Řeší chybu 'UnicodeEncodeError' nastavením výchozího kódování na UTF-8.
os.environ["PYTHONUTF8"] = "1"
# 2. Explicitně zakážeme trasování pro LangSmith, abychom se zbavili varování.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

# 3. Přidá aktuální adresář do cesty, kde Python hledá moduly.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Import pro Gemini model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
# Doporučený import pro Pydantic pro zajištění kompatibility
from pydantic import BaseModel, Field
# Přímý import pro Google AI
import google.generativeai as genai

# --- Krok 1: Import vašich existujících tříd agentů ---
from GetComodityPrice import WebCommodityAgent
from LoadClientsMultiFIleAgent import MultiExcelAgent, setup_dummy_excel_files

# --- Krok 2: Vytvoření "obálek" (nástrojů) pro supervisora ---

class CommodityPriceInput(BaseModel):
    pass # Tento nástroj nepotřebuje žádné argumenty

@tool(args_schema=CommodityPriceInput)
def commodity_price_tool() -> str:
    """
    Použij tento nástroj, pokud se uživatel ptá na aktuální cenu zlata,
    stříbra, platiny nebo Bitcoinu. Tento nástroj nepřijímá žádné argumenty.
    """
    print("\n>>> SUPERVISOR: Aktivuji agenta pro ceny komodit...")
    try:
        commodity_agent = WebCommodityAgent()
        prices = commodity_agent.get_commodity_prices()

        print(f"DEBUG: Agent vrátil surová data: {prices}")
        
        # Konverze numpy typů na standardní Python float
        if prices:
            for key, value in prices.items():
                if isinstance(value, dict):
                    prices[key]['USD'] = float(value.get('USD', 0.0))
                    prices[key]['CZK'] = float(value.get('CZK', 0.0))
        
        print(f"DEBUG: Data po konverzi na float: {prices}")
        
        if not prices or not any(prices.values()):
            return "Chyba: Agent pro komodity nevrátil žádná platná data. Možný problém s připojením k externím zdrojům (ECB, Yahoo Finance) nebo dočasný výpadek."

        output = "Zde jsou aktuální ceny:\n"
        for item, price_data in prices.items():
            if price_data:
                output += f"- {item}: {price_data['CZK']:,.2f} CZK\n"
            else:
                output += f"- {item}: Cena pro tuto položku není dostupná\n"
        return output

    except Exception as e:
        print("\n" + "!"*20 + " TRACEBACK CHYBY V NÁSTROJI " + "!"*20)
        traceback.print_exc()
        print("!"*60 + "\n")
        return f"Došlo k neočekávané interní chybě při volání agenta na ceny. Problém byl zaznamenán. Chyba: {e}"

# --- VYLEPŠENÍ AGENTA PRO KLIENTY ---
class ClientSearchInput(BaseModel):
    jmeno: Optional[str] = Field(default=None, description="Křestní jméno klienta pro vyhledání.")
    prijmeni: Optional[str] = Field(default=None, description="Příjmení klienta pro vyhledání.")
    mesto: Optional[str] = Field(default=None, description="Město, ve kterém se mají vyhledat klienti.")
    email: Optional[str] = Field(default=None, description="Email klienta pro vyhledání.")
    vek: Optional[int] = Field(default=None, description="Věk klienta pro vyhledání.")

@tool(args_schema=ClientSearchInput)
def client_data_tool(jmeno: Optional[str] = None, prijmeni: Optional[str] = None, mesto: Optional[str] = None, email: Optional[str] = None, vek: Optional[int] = None) -> str:
    """
    Použij tento nástroj pro vyhledávání klientských dat. Můžeš vyhledávat podle jména (jmeno), příjmení (prijmeni), města (mesto), emailu (email), věku (vek), nebo jejich kombinace.
    """
    print(f"\n>>> SUPERVISOR: Aktivuji agenta pro klientská data (Kritéria: jmeno={jmeno}, prijmeni={prijmeni}, mesto={mesto}, email={email}, vek={vek})...")
    
    kriteria = {}
    if jmeno:
        kriteria['Jméno'] = jmeno
    if prijmeni:
        kriteria['Příjmení'] = prijmeni
    if mesto:
        kriteria['Město'] = mesto
    if email:
        kriteria['Email'] = email
    if vek is not None:
        kriteria['Věk'] = vek

    if not kriteria:
        return "Je potřeba zadat alespoň jedno kritérium pro vyhledávání (jméno, příjmení, město, email, nebo věk)."

    file_paths = setup_dummy_excel_files()
    client_agent = MultiExcelAgent(file_paths=file_paths)
    found_clients = client_agent.najdi_vsechny_zaznamy(**kriteria)
    
    if not found_clients:
        return f"V databázi nebyli nalezeni žádní klienti odpovídající zadaným kritériím."
        
    output = f"Nalezení klienti:\n"
    for client in found_clients:
        output += f"- {client['Jméno']} {client['Příjmení']} (Email: {client['Email']}, Město: {client['Město']}, Věk: {client['Věk']})\n"
    return output

# --- Krok 3: Sestavení supervisora (bez spuštění) ---

# Zde vložte váš API klíč pro Google Gemini
GOOGLE_API_KEY = "AIzaSyBFvI7cR005hOH6mEO7ZwZT4prLWJ9xe4A"

def setup_supervisor():
    """Funkce pro sestavení a vrácení hotového supervisora."""
    tools = [commodity_price_tool, client_data_tool]
    
    # --- ZMĚNA: Explicitní vytvoření klienta ---
    # Tímto krokem obejdeme problematickou automatickou konfiguraci.
    # Ujistíme se, že se používá REST transport.
    client_options = {"api_key": GOOGLE_API_KEY}
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0,
        transport="rest",
        client_options=client_options
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Jsi nápomocný asistent. Pro odpověď na otázku můžeš použít dostupné nástroje."),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    supervisor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return supervisor

# --- Diagnostická funkce pro testování připojení ---
def test_gemini_connection():
    """Provede jednoduchý test připojení k Gemini API."""
    print("--- Provádím testovací volání Gemini API ---")
    try:
        # Použijeme přímou knihovnu google-generativeai pro nejčistší test
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content("Ahoj, funguješ?")
        print(f"✅ Testovací odpověď z Gemini: {response.text}")
        print("--- Testovací volání proběhlo úspěšně. ---")
        return True
    except Exception:
        print("\n" + "!"*60)
        print("!!! TEST PŘIPOJENÍ K GEMINI API SELHAL !!!")
        print("!"*60)
        print("I přímé volání API selhalo. Problém je pravděpodobně v konfiguraci.")
        print("\nProsím, zkontrolujte následující body:")
        print("1. API Klíč: Je klíč v proměnné GOOGLE_API_KEY vložen správně a bez překlepů?")
        print("2. Aktivované API: Je ve vašem Google Cloud projektu aktivováno 'Generative Language API' nebo 'Vertex AI API'?")
        print("3. Aktivovaný Billing: Je k vašemu Google Cloud projektu připojeno aktivní fakturační konto? (Toto je pro Gemini API vyžadováno).")
        print("\nDetailní traceback chyby:")
        traceback.print_exc()
        return False

# --- Krok 4: Vytvoření Flask aplikace ---

app = Flask(__name__)
CORS(app) # Povolí komunikaci mezi frontendem a backendem

# Sestavíme supervisora až poté, co ověříme připojení
supervisor_agent = None

@app.route('/chat', methods=['POST'])
def chat():
    """API endpoint pro chatování."""
    global supervisor_agent
    if supervisor_agent is None:
        print("Chyba: Supervisor není inicializován, protože test připojení selhal.")
        return jsonify({'error': 'Chyba serveru: Supervisor není k dispozici.'}), 500

    print("--- Nový dotaz na /chat endpoint ---")
    try:
        data = request.json
        user_input = data.get('message')
        print(f"Přijatý dotaz: {user_input}")
        
        if not user_input:
            print("Chyba: Chybí zpráva.")
            return jsonify({'error': 'Chybí zpráva (message)'}), 400
            
        result = supervisor_agent.invoke({"input": user_input})
        print(f"Supervisor vrátil výsledek: {result}")
        
        return jsonify({'reply': result['output']})

    except Exception as e:
        print(f"!!! CHYBA NA SERVERU !!!")
        traceback.print_exc()
        return jsonify({'error': 'Interní chyba serveru'}), 500

if __name__ == '__main__':
    # Nejprve otestujeme připojení
    if test_gemini_connection():
        # Pokud je připojení v pořádku, sestavíme supervisora a spustíme server
        print("\n--- Připojení k API je v pořádku. Sestavuji supervisora... ---")
        supervisor_agent = setup_supervisor()
        print("="*50)
        print("Spouštím Flask API server pro chatbota...")
        print("Backend bude dostupný na http://127.0.0.1:5000")
        print("="*50)
        app.run(host='0.0.0.0', port=5000, threaded=False)
    else:
        # Pokud test selže, server se nespustí
        print("\n--- Server se nespustí, dokud nebude opraven problém s připojením k API. ---")
        sys.exit(1)

