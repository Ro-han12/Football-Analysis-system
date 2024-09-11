from utils import read_video,save_video

def main():
    video_frames=read_video('input_videos/08fd33_4.mp4') #reading
    
    save_video(video_frames,'output_videos/output_video.avi') #saving
    
if __name__ == '__main__':
    main()