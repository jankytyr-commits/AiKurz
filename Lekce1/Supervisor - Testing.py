import os
import sys
import json
import codecs
import http.client # Import pro monkey-patching
import traceback # Import pro detailní výpis chyb

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
# ZMĚNA: Doporučený import pro Pydantic pro zajištění kompatibility
from langchain_core.pydantic_v1 import BaseModel, Field

# --- Krok 1: Import vašich existujících tříd agentů ---
# Díky úpravě sys.path výše by nyní měly být tyto importy bez chyby.
from GetComodityPrice import WebCommodityAgent
from LoadClientsMultiFIleAgent import MultiExcelAgent, setup_dummy_excel_files

# --- Krok 2: Vytvoření "obálek" (nástrojů) pro supervisora ---
# Tyto funkce budou volat instance vašich agentů.

# --- OPRAVA: Přidání prázdného vstupního schématu pro nástroj bez argumentů ---
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
        
        # Zpřesněná kontrola, zda byla vrácena nějaká platná data
        if not prices or not any(prices.values()):
            return "Chyba: Agent pro komodity nevrátil žádná platná data. Možný problém s připojením k externím zdrojům (ECB, Yahoo Finance) nebo dočasný výpadek."

        # Scénář 3: Vše proběhlo v pořádku, formátujeme výstup
        output = "Zde jsou aktuální ceny:\n"
        for item, price_data in prices.items():
            if price_data:
                output += f"- {item}: {price_data['CZK']:,.2f} CZK\n"
            else:
                output += f"- {item}: Cena pro tuto položku není dostupná\n"
        return output

    except Exception as e:
        print("\n" + "!"*20 + " TRACEBACK CHYBY V NÁSTROJI " + "!"*20)
        traceback.print_exc() # Vytiskne kompletní trasování chyby do konzole
        print("!"*60 + "\n")
        return f"Došlo k neočekávané interní chybě při volání agenta na ceny. Problém byl zaznamenán. Chyba: {e}"

# Definujeme vstupní schéma pro nástroj na klienty pro lepší kontrolu
class ClientSearchInput(BaseModel):
    city: str = Field(description="Město, ve kterém se mají vyhledat klienti.")

@tool(args_schema=ClientSearchInput)
def client_data_tool(city: str) -> str:
    """
    Použij tento nástroj pro vyhledávání klientských dat.
    Je nutné specifikovat město pro vyhledání.
    """
    print(f"\n>>> SUPERVISOR: Aktivuji agenta pro klientská data (Město: {city})...")
    
    # 1. Zajistíme, že existují testovací data
    file_paths = setup_dummy_excel_files()
    
    # 2. Vytvoříme instanci agenta pro klienty
    client_agent = MultiExcelAgent(file_paths=file_paths)
    
    # 3. Zavoláme metodu agenta pro nalezení klientů
    found_clients = client_agent.najdi_vsechny_zaznamy(Město=city)
    
    if not found_clients:
        return f"V databázi nebyli nalezeni žádní klienti z města {city}."
        
    # Převedeme seznam slovníků na přehledný text
    output = f"Nalezení klienti ve městě {city}:\n"
    for client in found_clients:
        output += f"- {client['Jméno']} {client['Příjmení']} (Email: {client['Email']})\n"
    return output

# --- Krok 3: Sestavení a spuštění supervisora ---

# Zde vložte váš API klíč pro Google Gemini
GOOGLE_API_KEY = "AIzaSyBFvI7cR005hOH6mEO7ZwZT4prLWJ9xe4A"

# Seznam všech nástrojů, které má supervisor k dispozici
tools = [commodity_price_tool, client_data_tool]

# Nastavení LLM, který bude fungovat jako mozek supervisora
# Přidán parametr transport="rest" pro obejití chyby s gRPC.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",
                             google_api_key=GOOGLE_API_KEY,
                             temperature=0,
                             transport="rest")

# --- ZMĚNA: Ruční vytvoření promptu místo stahování z hubu ---
# Tímto krokem se vyhýbáme potenciálním problémům se stahováním.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Jsi nápomocný asistent. Pro odpověď na otázku můžeš použít dostupné nástroje."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Sestavení jádra supervisora
agent = create_tool_calling_agent(llm, tools, prompt)

# Vytvoření exekutoru, který se stará o spuštění celého řetězce
supervisor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Krok 4: Otestování supervisora ---

print("\n" + "="*50)
print("SUPERVISOR JE PŘIPRAVEN PŘIJÍMAT POKYNY")
print("="*50)

# Test 1: Dotaz na cenu zlata
print("\n--- Test 1: Dotaz na cenu zlata ---")
result1 = supervisor.invoke({"input": "Jaká je teď prosím tě cena zlata?"})
print(f"\n✅ Finální odpověď supervisora:\n{result1['output']}")

print("\n" + "="*50)

# Test 2: Dotaz na klientská data
print("\n--- Test 2: Dotaz na klientská data ---")
result2 = supervisor.invoke({"input": "Potřeboval bych seznam všech našich klientů, kteří bydlí v Brně."})
print(f"\n✅ Finální odpověď supervisora:\n{result2['output']}")
