import time
import requests
import xml.etree.ElementTree as ET
import yfinance as yf

class WebCommodityAgent:
    """
    Agent, který získává aktuální ceny komodit a kryptoměn pomocí
    spolehlivých finančních dat (yfinance a ECB).
    """

    def __init__(self):
        """Inicializace agenta."""
        print("Agent pro sledování komodit je aktivní.")
        
        # Konfigurace symbolů pro Yahoo Finance
        self.asset_config = {
            'Zlato (GLD ETF)': 'GLD',
            'Stříbro (SLV ETF)': 'SLV',
            'Platina (PPLT ETF)': 'PPLT',
            'Bitcoin': 'BTC-USD'
        }

    def _get_usd_to_czk_rate_from_ecb(self):
        """
        Získá aktuální kurz USD/CZK z denního XML feedu Evropské centrální banky.
        Kurz je vypočítán přes Euro jako základní měnu.
        """
        try:
            print("  -> Stahuji kurzovní lístek z Evropské centrální banky...")
            url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            tree = ET.fromstring(response.content)

            namespaces = {'def': 'http://www.ecb.int/vocabulary/2002-08-01/eurofxref'}
            
            # Najde kontejner s kurzy
            cube = tree.find('.//def:Cube/def:Cube', namespaces)
            if cube is None:
                return None

            rates = {rate.attrib['currency']: float(rate.attrib['rate']) 
                     for rate in cube.findall('def:Cube', namespaces)}

            if 'USD' in rates and 'CZK' in rates:
                # Výpočet křížového kurzu CZK/USD
                return rates['CZK'] / rates['USD']
            return None
        except requests.RequestException as e:
            print(f"  -> Chyba při stahování dat z ECB: {e}")
            return None
        except ET.ParseError as e:
            print(f"  -> Chyba při zpracování XML od ECB: {e}")
            return None


    def get_commodity_prices(self):
        """
        Získá ceny pro všechny nakonfigurované komodity.
        
        Vrací:
            dict: Slovník s cenami komodit v USD a CZK.
        """
        print("\nAgent se připojuje k datovým zdrojům...")
        
        usd_czk_rate = self._get_usd_to_czk_rate_from_ecb()
        if usd_czk_rate is None:
            print("  -> Kritická chyba: Nepodařilo se získat kurz USD/CZK. Ceny v CZK nebudou dostupné.")
            return None

        print(f"  -> Aktuální kurz USD/CZK podle ECB: {usd_czk_rate:.4f}")
        
        results = {}
        for name, symbol in self.asset_config.items():
            print(f"  -> Získávám data pro: {name} ({symbol})")
            try:
                ticker = yf.Ticker(symbol)
                # Získáme poslední dostupná data
                data = ticker.history(period='5d', interval='1d')
                
                if not data.empty:
                    price_usd = data['Close'].iloc[-1]
                    price_czk = price_usd * usd_czk_rate
                    results[name] = {'USD': price_usd, 'CZK': price_czk}
                else:
                    print(f"  -> Varování: Pro symbol '{symbol}' nebyla nalezena žádná data.")
                    results[name] = None
            except Exception as e:
                print(f"  -> Chyba při získávání dat pro '{symbol}': {e}")
                results[name] = None
            time.sleep(0.2) # Malá pauza mezi dotazy

        print("\nAgent úspěšně dokončil získávání dat.")
        return results

# --- Příklad použití agenta ---
if __name__ == "__main__":
    # 1. Vytvoření instance agenta
    commodity_agent = WebCommodityAgent()

    # 2. Získání dat od agenta
    current_prices = commodity_agent.get_commodity_prices()

    # 3. Zobrazení výsledků
    print("\n" + "="*55)
    print("AKTUÁLNÍ PŘEHLED CEN KOMODIT")
    print("="*55)
    
    if current_prices:
        for item, prices in current_prices.items():
            if prices:
                # Formátování pro lepší čitelnost čísel
                print(f"{item:<25}: {prices['USD']:>12,.2f} USD / {prices['CZK']:>12,.2f} CZK")
            else:
                print(f"{item:<25}: Cena nebyla dostupná")
    else:
        print("Nepodařilo se získat žádná data.")
        
    print("="*55)
