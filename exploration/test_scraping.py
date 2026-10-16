import requests
from bs4 import BeautifulSoup

url = "https://www.airlinequality.com/airline-reviews/ryanair/"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

risposta = requests.get(url, headers=headers)
print("Codice di risposta:", risposta.status_code)
print("Lunghezza:", len(risposta.text), "caratteri")

soup = BeautifulSoup(risposta.text, "html.parser")
titolo = soup.find("title")
print("Titolo della pagina:", titolo.get_text() if titolo else "nessun titolo")

recensioni = soup.find_all("div", class_="text_content")
print("Recensioni trovate:", len(recensioni))