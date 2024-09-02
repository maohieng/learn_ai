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
