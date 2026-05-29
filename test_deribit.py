import requests

r = requests.get(
    "https://www.deribit.com/api/v2/public/get_instruments",
    params={"currency": "ETH", "kind": "option", "expired": False},
    timeout=10
)
data = r.json()
instruments = data["result"]
print(f"Toplam ETH option: {len(instruments)}")
print(f"Ornek: {instruments[0]['instrument_name']}")
print(f"Strike: {instruments[0]['strike']}")
print(f"Expiry: {instruments[0]['expiration_timestamp']}")
