import logging
from dao import JsonFileDao
from song_album import Album

class AlbumFileDao(JsonFileDao):

    def __init__(self, jsonfile):
        super().__init__(filename=jsonfile, id_fieldname='ID')

    def store_one(self, data):
        logging.info(f"AlbumRepo: storing {data}")
        super().store_one(data)

    def store_all(self, datas):
        logging.info(f"AlbumRepo: storing all")
        return super().store_all(datas)

    def fetch(self):
        logging.info(f"AlbumRepo: fetching")
        # fetch from json file
        data = super().fetch()
        return [Album.from_dict(d) for d in data]
        

    def delete(self, id):
        logging.info(f"AlbumRepo: deleting {id}")
        super().delete(id)
        