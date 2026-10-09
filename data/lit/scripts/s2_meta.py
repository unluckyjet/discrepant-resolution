import json, os, sys, time, urllib.request
ids = sys.argv[1:]
key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
for pid in ids:
    url = f"https://api.semanticscholar.org/graph/v1/paper/{pid}?fields=title,year,venue,externalIds,publicationVenue"
    req = urllib.request.Request(url, headers={"x-api-key": key} if key else {})
    try:
        d = json.load(urllib.request.urlopen(req))
        print(json.dumps({"id": pid, "title": d.get("title"), "year": d.get("year"), "venue": d.get("venue"), "ext": d.get("externalIds")}))
    except Exception as e:
        print(json.dumps({"id": pid, "error": str(e)}))
    time.sleep(1.2)
