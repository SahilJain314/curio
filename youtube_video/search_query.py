"""
This script is used to query the Youtube API using the python-youtube
package in order to retrieve all relevant youtube data and stores
it in a json.

Make sure you have downloaded the required packages by doing:
pip install --upgrade python-youtube
pip install youtube_transcript_api
"""

from pyyoutube import Api
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_channels import Channel
import json
import copy


class Video_Search_Json:
    """Class to retrieve videos

    This class helps query and search our video database
    using an API YouTube key if needed to pull from Youtube.
    Automatically updates the json file.
    """
    
    def __init__(self, API_KEY, JSON_PATH):
        # Declare an API object using our API_KEY
        self.api = Api(api_key=API_KEY)
        self.data_path = JSON_PATH

        with open(JSON_PATH) as f:
            self.database = json.load(f)


    def search_by_keywords(self, subtopic, channels=[Channel['MIT']], num_videos=10, force_query=False, include_transcripts=True):
        """Searches for videos by subtopic

        Takes in a query string to search for on youtube.
        Returns a JSON.

        Parameters:
        subtopic -- the subtopic to search for in the query
        channels -- a list of Channel enums to specify which channels that must be included (default MIT OpenCourseWare)
        num_vidoes -- minimum number of videos to include (default 10)
        force_query -- whether or not to query regardless of inclusion in json (default False)
        include_transcripts -- whether or not to include transcripts (default True)
        """
        
        if subtopic not in self.database.keys() and not force_query:
            # YouTube retrieve
            self.query_youtube(subtopic, channels, num_videos=num_videos, include_transcripts=include_transcripts)

            self.write_to_json(self.database)
        return self.database[subtopic]
    
    
    def query_youtube(self, subtopic, channels=[], num_videos=5, include_transcripts=True, search_count=50):
        """Query Youtube for a subtopic
        
        Queries the YouTube database for 5 videos pertaining to a certain subtopic. 
        Automatically re-queries if a video doesn't include a transcript if transcripts
        are required.
        
        Parameters:
        subtopic -- the topic to search for
        channels -- specifically which Channel enums to also include from the Channels class (default empty)
        num_videos -- the number of videos to return. Could be more if a channels argument is non-empty (default 5)
        include_transcripts -- requires videos to have transcripts (default True)
        count -- the number if videos to query for everytime (default 50)
        """
        assert num_videos >= 0 # Number of videos cannot be negative

        # Query Youtube and add videos pertaining to the subtopic
        r = self.api.search_by_keywords(q=subtopic, search_type=["video"], count=search_count, limit=search_count, video_caption="any", video_duration=["any"])
        
        # Add list of videos to database
        videos = []

        # Creates deep copy of Channel(s) to keep track of which ones are included
        includes_channels = []
        for channel_enum in channels:
            includes_channels.append(channel_enum.name)

        # Maximum amount of videos to include (one video from each channel plus top 5 from results)
        video_counter = num_videos

        for vid in r.items:
            should_append = False

            # Check to see if max number of videos has been reached
            if video_counter <= 0 and len(includes_channels) == 0:
                break

            # Filter the video from the YouTube API
            filtered_video = self.filter_video_information(vid.to_dict())

            # Conditions for when to add a video
            if filtered_video["channelId"] in includes_channels:
                # Remove minimum one video from channel requirement if video found
                includes_channels.remove(filtered_video["channelId"])
                should_append = True
            elif video_counter > 0:
                # Add video if we still need to add videos to reach minimum number of videos
                should_append = True
            
            if not should_append:
                continue
            
            # Include transcripts if specified
            if include_transcripts and should_append:
                filtered_video["transcript"] = self.get_youtube_transcript(filtered_video["videoId"])
                if filtered_video["transcript"] == None:
                    continue
            
            if should_append:
                # Add in other fields
                filtered_video["url"] = "www.youtube.com/watch?v=" + filtered_video["videoId"]
                filtered_video["source"] = "Youtube"
                filtered_video["difficulty"] = 3 # Default difficulty level

                # Add video to list
                videos.append(filtered_video)
                video_counter -= 1 # Decrement video_counter for minimum number of videos to include

        # Add filtered videos into the database (mutates)
        self.database[subtopic] = videos
    

    def filter_video_information(self, video, keys=["publishedAt", "channelId", "title", "description", "channelTitle", "videoId"]):
        """Filters video dict for certain keys

        Filters a YouTube Video entry to only include a certain number of keys
        specified by a keys list taken from the YouTube API.

        Parameters:
        video -- the video information as a dictionary to filter through
        keys -- the keys to include (default ["publishedAt", "channelId", "title", "description", "channelTitle", "videoId])
        """

        new_video = {}
        self.recur_dict(video, new_video, keys) # Recursively loop through nested dictionaries and put everything on first layer
        return new_video


    def get_youtube_transcript(self, video_id):
        """Returns video's transcript from YouTube

        Returns the video's transcripts given the video_id on YouTube. Returns
        None if no transcript was found. This functionality is included in order
        to check whether or not a video holds a transcript.

        Parameters:
        video_id -- the id of the video on YouTube. Can be found after the "v=" part in the link.
        """
        
        # Try grabbing the raw translation using the YouTubeTranscriptApi
        raw_trans = []
        try:
            raw_trans = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        except (Exception):
            print(video_id, "excluded")
            return None
        
        # Use only the text portion of transcript and throwaway time
        transcript = ""
        for i in raw_trans:
            transcript += i["text"] + " "
        return transcript


    def recur_dict(self, data, output, keys_to_include):
        """Recursively loop through nested dict

        Recursive function for reading nested dictionaries and 
        retrieving the keys to a one-layer dictionary.
        """

        # Loop through key value pair in dictionary
        for key, value in data.items():
            if isinstance(value, dict):
                # If the value is a dict, recursively loop
                self.recur_dict(value, output, keys_to_include)
            elif key in keys_to_include:
                # If value is not a dictionary, add to new dict
                output[key] = value

    
    def write_to_json(self, data_dict):
        """Write dictionary to JSON file

        Writes the file into a JSON and includes Exception
        protection and null dictionary protection.
        
        Parameters:
        data_dict -- the dictionary to write to json
        """

        try:
            # Only write data to JSON if it is non Null
            if data_dict:
                with open(self.data_path, 'w') as json_file:
                    json.dump(data_dict, json_file)
        except (json.decoder.JSONDecodeError):
            print("Error Writing to Json File, Dictionary improperly formatted.")


    