from typing import List
from song_album import Album
from song import Song

dspFmtWithNum = '| {0:^3} | {1:18} | {2:12} | {3:5} | {4:26} |'

def menu():
    print("====== Menu ======")
    print("1. View a music store")
    print("2. Add a song")
    print("3. Create an album")
    print("4. Quit")
    print("==================")
    return int(input("Enter your choice: "))

def selectAlbum(deps: List[Album]) -> Album:
    print("======= Select an album =======")
    for i, dep in enumerate(deps):
        print(f"{i+1}. {dep.title}")
    while True:
        choice = int(input("Enter your choice: "))
        if choice < 1 or choice > len(deps):
            print("Invalid choice!")
            continue
        else:
            return deps[choice-1]
    

def createSong(deps: List[Album]) -> Song:
    # Select an album
    album = selectAlbum(deps)
    print("======= Add a song =======")
    title = input("Enter title: ")
    artist = input("Enter artist: ")
    user_input_track = input("Enter track number (auto): ")
    track_number = 0
    if user_input_track.strip() == "":
        track_number = 0
    else:
        track_number = int(user_input_track)
        
    song = Song(title, artist, track_number)

    # Add song to album
    album.add_song(song)
    return song

def listSongs(songs: List[Song]):
    if len(songs) == 0:
        print("< No songs in this album >")
        return

    for i, song in enumerate(songs):
        print(dspFmtWithNum.format(i+1, song.title, song.artist, song.track_number, str(song.ID)))

def createAlbum() -> Album:
    print("======= Create an album =======")
    title = input("Enter title: ")
    genre = input("Enter genre: ")
    year = int(input("Enter year: "))
    return Album(title, genre, year)

def main(deps: List[Album] =[]):
    while True:
        choice = menu()
        if choice == 1: # View Music Store
            for dep in deps:
                # print header
                print(f'==================== Album: {dep.title} ({dep.genre}) {dep.year} ====================')
                print(dspFmtWithNum.format("No.", "Title", "Artist", "Track", "ULID"))
                print('===============================================================================')
                # print data
                listSongs(dep.songs)
                print('===============================================================================')
                print()
        elif choice == 2: # Add a song
            createSong(deps=deps)
            print("Song added!")
            print()
        elif choice == 3: # Create an album
            print("========= Create an album =========")
            deps.append(createAlbum())
            print("Album added!")
            print()
        elif choice == 4: # Quit
            print("========= Quit =========")
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")
            continue    

if __name__ == '__main__':
    deps = [
        Album("Skull 1", "Hiphop", 2021),
        Album("Reborn", "Rock", 2020),
        Album("THINK LATER", "Pop", 2023)
    ]

    deps[0].add_song(Song("Solo", "Vannda"))

    deps[1].add_song(Song("How about now", "G-Davit"))

    deps[2].add_song(Song("greedy", "Tate McRae"))

    main(deps=deps)