import logging
from dao import TextAndJsonFileDao
from video import Video

class VideoFileDao(TextAndJsonFileDao):

    def __init__(self, textfile, jsonfile):
        super().__init__(filename=textfile, jsonfile=jsonfile)

    def data_to_text(self, data):
        return f"{data.type},{data.title},{data.length},{data.artist}"

    def text_to_data(self, text):
        lines = text.split('\n')
        videos = []
        for line in lines:
            if line:
                type, title, length, artist = line.split(',')
                s = Video(type=type, title=title, length=int(length), artist=artist)
                videos.append(s)
        return videos

    def store_one(self, data):
        logging.info(f"VideoRepo: storing {data}")
        super().store_one(data)

    def store_all(self, datas):
        logging.info(f"VideoRepo: storing all")
        return super().store_all(datas)

    def fetch(self):
        logging.info(f"VideoRepo: fetching")
        # return super().fetch()
        # fetch from json file
        data = self.jsonDao.fetch()
        return [Video.from_dict(d) for d in data]
        

    def delete(self, id):
        logging.info(f"VideoRepo: deleting {id}")
        pass
        