import requests

url = "https://nd0821-c3-starter-code-master-76k0.onrender.com/inference"

payload = {
    "age": 42,
    "workclass": "Private",
    "education": "Bachelors",
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "hours-per-week": 40,
    "native-country": "United-States"
}

response = requests.post(url, json=payload)
print(f"Status code: {response.status_code}")
print(f"Result: {response.json()}")