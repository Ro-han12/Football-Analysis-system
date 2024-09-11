from utils import read_video,save_video
from trackers import Tracker


def main():
    video_frames=read_video('input_videos/08fd33_4.mp4') #reading
    
    tracker=Tracker('models/best.pt') #initialize tracker
    tracks=tracker.get_object_tracks(video_frames)
    
    save_video(video_frames,'output_videos/output_video.avi') #saving
    
if __name__ == '__main__':
    main()