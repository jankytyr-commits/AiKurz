import yfinance as yf
import requests
import xml.etree.ElementTree as ET

def get_usd_to_czk_rate_ecb():
    url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
    response = requests.get(url)
    tree = ET.fromstring(response.content)

    namespaces = {
        'gesmes': 'http://www.gesmes.org/xml/2002-08-01',
        'def': 'http://www.ecb.int/vocabulary/2002-08-01/eurofxref'
    }

    cube = tree.find('.//def:Cube/def:Cube', namespaces)
    rates = {}
    for rate in cube.findall('def:Cube', namespaces):
        currency = rate.attrib['currency']
        value = float(rate.attrib['rate'])
        rates[currency] = value

    if 'USD' in rates and 'CZK' in rates:
        return rates['CZK'] / rates['USD']
    else:
        return None

symbols = {
    'Zlato (GLD ETF – USD)': 'GLD',
    'Stříbro (SLV ETF – USD)': 'SLV',
    'Platina (PPLT ETF – USD)': 'PPLT',
    'Bitcoin (BTC-USD)': 'BTC-USD'
}

usd_czk = get_usd_to_czk_rate_ecb()
if usd_czk is None:
    print("Nepodařilo se načíst kurz USD/CZK z ECB.")
else:
    print(f"Aktuální kurz USD/CZK podle ECB: {usd_czk:.4f}")
    print("="*50)

    for name, sym in symbols.items():
        ticker = yf.Ticker(sym)
        data = ticker.history(period='1d')
        if not data.empty:
            price_usd = data['Close'].iloc[-1]
            price_czk = price_usd * usd_czk
            print(f"{name:35}: {price_usd:.2f} USD / {price_czk:.2f} CZK")
        else:
            print(f"{name:35}: Cena nebyla dostupná")
