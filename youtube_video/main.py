from search_query import Video_Search_Json
from youtube_channels import Channel
import json

# API Key from Youtube API for accessing Youtube's public data
API_KEY = "AIzaSyAQhD0w3nsWnDYAebl6TAzZuqSjdvQtlIc" # does key reset at 2am? yes 24
# API_KEY = "AIzaSyDD_NVaMnF9MgTDsS9DtEMT_2Fz60_-NWM" # <- works 3
# API_KEY = "zerotwo"

JSON_PATH = "full_data.json"

vsj = Video_Search_Json(API_KEY, JSON_PATH)

# Main method
if __name__ == "__main__":
    with open("syllabi.json") as f:
        topics = json.load(f)

    for topic in topics:
        for subtopic in topics[topic]:
            vsj.search_by_keywords(subtopic + " " + topic, include_transcripts=True)
    
    # vsj.search_by_keywords("heaps", include_transcripts=False)
    print("done")