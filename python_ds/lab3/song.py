from ulid import ULID

class Song:
    '''A song in an album.'''
    '''The ID will be automatically generated.'''
    ID: ULID
    title: str
    
    '''The artist of the song.'''
    artist: str

    '''The track number started from 1. It will be automatically assigned if it is not provided.'''
    track_number: int
    
    # '''The album that this song belongs to.'''
    # album: ULID

    def __init__(self, title, artist, track_number=0):
        self.title = title
        self.artist = artist
        self.track_number = track_number
        self.ID = ULID()

    def __str__(self):
        return f'{self.title} by {self.artist}'

    def __eq__(self, value: object) -> bool:
        return self.ID.__eq__(value.ID)
    
    def __hash__(self) -> int:
        return self.ID.__hash__()
    
    # def __lt__(self, value: object) -> bool:
    #     return self.track_number < value.track_number
    
    # def __gt__(self, value: object) -> bool:
    #     return self.track_number > value.track_number

    def to_dict(self):
        return {
            'ID': str(self.ID),
            'title': self.title,
            'artist': self.artist,
            'track_number': self.track_number
        }

    @staticmethod
    def from_dict(data):
        s = Song(title=data['title'], artist=data['artist'], track_number=data['track_number'])
        s.ID = ULID.from_str(data['ID'])
        return s