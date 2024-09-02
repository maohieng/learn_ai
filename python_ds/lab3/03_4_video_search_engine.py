from typing import List
from video_dao import VideoFileDao
from video import Video

dspFmtWithNum = '| {0:^3} | {1:^4} | {2:28} | {3:16} | {4:^6} |'

videoDao = VideoFileDao('videos.txt', 'videos.json')

def menu():
    print("====== Menu ======")
    print("1. Search")
    print("2. View all videos")
    print("3. Add a new video")
    print("4. Quit")
    print("==================")
    return int(input("Enter your choice: "))

def createVideo():
    name = input("Enter title: ")
    length = int(input("Enter length: "))
    type = input("Enter type (mp4, mkv, etc.): ")
    artist = input("Enter artist: ")
    videoDao.store_one(Video(title=name, length=length, artist=artist, type=type))

def listVideos():
    # Fetch videos
    videos = videoDao.fetch()
    for i, b in enumerate(videos):
        print(dspFmtWithNum.format(i+1, b.type, b.title, 'unknown' if not b.artist else b.artist, b.length))

def searchVideo(s: str) -> List[Video]:
    videos = videoDao.fetch()
    # Search in title
    foundVid = list(filter(lambda v: s.lower() in v.title.lower(), videos))
    # Search in artist
    foundVid += list(filter(lambda v: s.lower() in v.artist.lower(), videos))
    return foundVid

def main():
    while True:
        choice = menu()
        if choice == 1: # Search
            print("====== Search video ======")
            s = input("Search anything: ")    
            fv = searchVideo(s)
            print()
            if len(fv) == 0:
                print("No video found!")
            else:
                print(f"{len(fv)} videos found:")
                for i, v in enumerate(fv):
                    print(f"{i+1}. {v}")
            print()
        elif choice == 2: # View all
            print("====== View all videos ======")
            # print header
            print('==============================================================================')
            print(dspFmtWithNum.format("No.", "Type", "Title", "Artist", "Length"))
            print('==============================================================================')
            # print data
            listVideos()
            print('==============================================================================')
            print()
        elif choice == 3: # Add new
            print("====== Add a new video ======")
            try:
                createVideo()
            except ValueError as e:
                print(e)
                print()
                continue
            print("video added!")
            print()
        elif choice == 4: # Quit
            print("====== Quit ======")
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")
            continue    

if __name__ == '__main__':
    # initialize some data for testing
    try:
        videoDao.fetch()
    except FileNotFoundError:
        videos = [
            Video(type="mp4", title="Prey Eh Kert", length=120, artist="sin sisamuth"),
            Video(type="mp4", title="Chnam Oun 16", length=180, artist="Sinn Sisamouth"),
            Video(type="mkv", title="Luoch Sneh Luoch Tuk", length=150, artist="Sinn Sisamouth"),
            Video(type="mp4", title="Kampong Saom", length=200, artist="Ros Sereysothea"),
            Video(type="mp4", title="បុប្ផាភូមិបាក់ព្រា", length=200, artist="រស់ សេរីសុទ្ធា"),
        ]
        videoDao.store_all(videos)
    
    main()