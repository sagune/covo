import requests


urls = [
    "https://hf-mirror.com/datasets-server/filter?dataset=parler-tts/mls_eng&config=default&split=train&where=%22speaker_id%22=%278576%27&length=1",
    "https://hf-mirror.com/api/datasets/parler-tts/mls_eng/parquet/default/train",
    "https://r.jina.ai/http://datasets-server.huggingface.co/filter?dataset=parler-tts/mls_eng%26config=default%26split=train%26where=%22speaker_id%22=%278576%27%26length=1",
]
for url in urls:
    try:
        response = requests.get(url, timeout=30)
        print(url, response.status_code, response.text[:700])
    except Exception as exc:
        print(url, type(exc).__name__, exc)
