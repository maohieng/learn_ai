class Video:
    type: str
    title: str
    length: int
    artist: str

    # Default value of parameter does not work in case user uses keyword argument
    def __init__(self, title: str, length: int, artist: str, type='mp4'):
        self.title = title
        self.length = length
        self.artist = artist
        self.type = type
    
    def __str__(self):
        return self.title + " by " + self.artist

    def to_dict(self):
        return {
            'type': self.type,
            'title': self.title,
            'length': self.length,
            'artist': self.artist
        }
    
    @staticmethod
    def from_dict(data: dict):
        return Video(type=data['type'], title=data['title'], length=data['length'], artist=data['artist'])
    