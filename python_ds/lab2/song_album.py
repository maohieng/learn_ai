from ulid import ULID
from song import Song

class Album:

    def __init__(self, title, genre, year, songs=None):
        '''
        **Python mutable default arguments encountered!!**

        **Issue**: Using songs=[] as a default argument means that all instances 
        of Album will share the same list if no songs argument is provided.
        
        **Solution**: Use songs=None as a default argument and create a new list
        if songs is None.
        '''
        self.title = title
        self.genre = genre
        self.year = year
        self.songs = songs if songs is not None else []
        self.ID = ULID()

    def add_song(self, song: Song):
        # song.album = self.ID
        if song.track_number == 0:
            song.track_number = len(self.songs) + 1
        
        self.songs.append(song)

    def __str__(self):
        return f'{self.title} - {self.genre} ({self.year})'