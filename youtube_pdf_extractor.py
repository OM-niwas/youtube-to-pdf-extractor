"""
YouTube Video to PDF Frame Extractor
====================================
This script downloads YouTube videos (single or playlist) and extracts unique frames,
then converts them into a PDF with timestamps for easy reference.

Perfect for creating study materials, presentations, or documentation from video content!
"""

# Import all necessary libraries for our video processing magic
import sys
from PIL import ImageFile
sys.modules['ImageFile'] = ImageFile  # Fix for PIL import issues

import cv2                          # For video processing and frame extraction
import os                           # For file and directory operations
import tempfile                     # For creating temporary folders
import re                           # For pattern matching in URLs
from fpdf import FPDF              # For creating PDF documents
from PIL import Image              # For image processing
import yt_dlp                      # For downloading YouTube videos
from skimage.metrics import structural_similarity as ssim  # For comparing frame similarity
from scipy.spatial import distance  # For mathematical distance calculations


def download_video(url, filename, max_retries=3):
    """
    Downloads a YouTube video with retry mechanism
    
    Args:
        url (str): The YouTube video URL to download
        filename (str): What to name the downloaded file
        max_retries (int): How many times to retry if download fails
    
    Returns:
        str: The filename of the downloaded video
    
    Raises:
        Exception: If all download attempts fail
    """
    # Configure yt-dlp with our download preferences
    ydl_opts = {
        'outtmpl': filename,    # Template for output filename
        'format': 'best',       # Download the best quality available
    }
    
    retries = 0  # Keep track of how many times we've tried
    
    # Keep trying until we succeed or run out of attempts
    while retries < max_retries:
        try:
            # Create a YouTube downloader with our options
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])  # Download the video
            return filename  # Success! Return the filename
            
        except yt_dlp.utils.DownloadError as e:
            # Something went wrong, let's try again
            print(f"Error downloading video: {e}. Retrying... (Attempt {retries + 1}/{max_retries})")
            retries += 1
    
    # If we get here, all attempts failed
    raise Exception("Failed to download video after multiple attempts.")


def get_video_id(url):
    """
    Extracts the unique video ID from various YouTube URL formats
    
    This function can handle:
    - YouTube Shorts: youtube.com/shorts/VIDEO_ID
    - Short URLs: youtu.be/VIDEO_ID
    - Regular URLs: youtube.com/watch?v=VIDEO_ID
    - Live streams: youtube.com/live/VIDEO_ID
    
    Args:
        url (str): The YouTube URL to parse
    
    Returns:
        str: The video ID if found, None otherwise
    """
    # Try to match YouTube Shorts URLs (youtube.com/shorts/VIDEO_ID)
    video_id_match = re.search(r"shorts/(\w+)", url)
    if video_id_match:
        return video_id_match.group(1)

    # Try to match shortened YouTube URLs (youtu.be/VIDEO_ID)
    video_id_match = re.search(r"youtu\.be\/([\w\-_]+)(\?.*)?", url)
    if video_id_match:
        return video_id_match.group(1)

    # Try to match regular YouTube URLs (youtube.com/watch?v=VIDEO_ID)
    video_id_match = re.search(r"v=([\w\-_]+)", url)
    if video_id_match:
        return video_id_match.group(1)

    # Try to match YouTube live stream URLs (youtube.com/live/VIDEO_ID)
    video_id_match = re.search(r"live\/(\w+)", url)  
    if video_id_match:
        return video_id_match.group(1)

    # If none of the patterns match, we couldn't find a video ID
    return None


def get_playlist_videos(playlist_url):
    """
    Extracts all video URLs from a YouTube playlist
    
    Args:
        playlist_url (str): The YouTube playlist URL
    
    Returns:
        list: A list of individual video URLs from the playlist
    """
    # Configure yt-dlp for playlist extraction (without downloading)
    ydl_opts = {
        'ignoreerrors': True,      # Skip videos that can't be processed
        'playlistend': 1000,       # Maximum number of videos to fetch
        'extract_flat': True,      # Just get URLs, don't download metadata
    }
    
    # Extract playlist information
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        playlist_info = ydl.extract_info(playlist_url, download=False)
        # Return a list of all video URLs in the playlist
        return [entry['url'] for entry in playlist_info['entries']]


def extract_unique_frames(video_file, output_folder, n=3, ssim_threshold=0.8):
    """
    Extracts unique frames from a video by comparing similarity between frames
    
    This function is smart - it only saves frames that are significantly different
    from the previous ones, avoiding duplicate slides or static content.
    
    Args:
        video_file (str): Path to the video file
        output_folder (str): Where to save the extracted frames
        n (int): Process every nth frame (3 = every 3rd frame for speed)
        ssim_threshold (float): Similarity threshold (0.8 = 80% similar frames are skipped)
    
    Returns:
        list: List of tuples containing (frame_number, timestamp_in_seconds)
    """
    # Open the video file for processing
    cap = cv2.VideoCapture(video_file)
    
    # Get the frames per second of the video (needed for timestamp calculation)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize variables for frame comparison
    last_frame = None                    # Previous frame for comparison
    saved_frame = None                   # Current frame to potentially save
    frame_number = 0                     # Current position in video
    last_saved_frame_number = -1         # When did we last save a frame?
    timestamps = []                      # List to store frame info
    
    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()  # Read the next frame
        
        if not ret:  # If we can't read more frames, we're done
            break
        
        # Only process every nth frame to speed up processing
        if frame_number % n == 0:
            # Convert to grayscale and resize for faster comparison
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, (128, 72))  # Small size for speed
            
            # If we have a previous frame to compare with
            if last_frame is not None:
                # Calculate how similar this frame is to the last one
                similarity = ssim(gray_frame, last_frame, 
                                data_range=gray_frame.max() - gray_frame.min())
                
                # If the frames are different enough (below threshold)
                if similarity < ssim_threshold:
                    # Make sure enough time has passed since last save (avoid rapid saves)
                    if saved_frame is not None and frame_number - last_saved_frame_number > fps:
                        # Create filename with frame number and timestamp
                        frame_path = os.path.join(output_folder, 
                                                f'frame{frame_number:04d}_{frame_number // fps}.png')
                        # Save the frame as an image
                        cv2.imwrite(frame_path, saved_frame)
                        # Record when we saved this frame
                        timestamps.append((frame_number, frame_number // fps))
                    
                    # Update our saved frame and tracking
                    saved_frame = frame
                    last_saved_frame_number = frame_number
                else:
                    # Frames are too similar, but keep the current frame as potential save
                    saved_frame = frame
            
            else:
                # This is the very first frame we're processing - always save it
                frame_path = os.path.join(output_folder, 
                                        f'frame{frame_number:04d}_{frame_number // fps}.png')
                cv2.imwrite(frame_path, frame)
                timestamps.append((frame_number, frame_number // fps))
                last_saved_frame_number = frame_number
            
            # Remember this frame for next comparison
            last_frame = gray_frame
        
        frame_number += 1  # Move to next frame
    
    # Clean up video capture
    cap.release()
    return timestamps


def convert_frames_to_pdf(input_folder, output_file, timestamps):
    """
    Converts extracted frames into a beautiful PDF with timestamps
    
    Each frame becomes a page in the PDF with a timestamp overlay
    that automatically adjusts color based on the background.
    
    Args:
        input_folder (str): Folder containing the frame images
        output_file (str): Name of the output PDF file
        timestamps (list): List of (frame_number, timestamp_seconds) tuples
    """
    # Get all frame files and sort them by frame number
    frame_files = sorted(os.listdir(input_folder), 
                        key=lambda x: int(x.split('_')[0].split('frame')[-1]))
    
    # Create a PDF in landscape orientation (better for video frames)
    pdf = FPDF("L")  # "L" = Landscape
    pdf.set_auto_page_break(0)  # Don't automatically break pages
    
    # Process each frame and add it to the PDF
    for i, (frame_file, (frame_number, timestamp_seconds)) in enumerate(zip(frame_files, timestamps)):
        frame_path = os.path.join(input_folder, frame_file)
        image = Image.open(frame_path)  # Open the frame image
        
        # Add a new page to the PDF
        pdf.add_page()
        
        # Add the frame image to fill the entire page
        pdf.image(frame_path, x=0, y=0, w=pdf.w, h=pdf.h)
        
        # Convert seconds to HH:MM:SS format for better readability
        timestamp = f"{timestamp_seconds // 3600:02d}:{(timestamp_seconds % 3600) // 60:02d}:{timestamp_seconds % 60:02d}"
        
        # Smart timestamp color selection based on background
        # We sample a small area where the timestamp will be placed
        x, y, width, height = 5, 5, 60, 15  # Top-left corner area
        region = image.crop((x, y, x + width, y + height)).convert("L")  # Convert to grayscale
        mean_pixel_value = region.resize((1, 1)).getpixel((0, 0))  # Average brightness
        
        # Choose text color based on background brightness
        if mean_pixel_value < 64:  # Dark background
            pdf.set_text_color(255, 255, 255)  # White text
        else:  # Light background
            pdf.set_text_color(0, 0, 0)      # Black text
        
        # Position and add the timestamp text
        pdf.set_xy(x, y)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 0, timestamp)  # Add the timestamp text
    
    # Save the completed PDF
    pdf.output(output_file)


def get_video_title(url):
    """
    Gets the title of a YouTube video and cleans it for use as filename
    
    Args:
        url (str): The YouTube video URL
    
    Returns:
        str: Clean video title suitable for use as filename
    """
    # Configure yt-dlp to only get video info (no download)
    ydl_opts = {
        'skip_download': True,    # Don't download the video
        'ignoreerrors': True      # Don't crash on errors
    }
    
    # Extract video information
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        video_info = ydl.extract_info(url, download=False)
        
        # Get the title and clean it up for filename use
        # Replace characters that aren't allowed in filenames
        title = video_info['title'].replace('/', '-').replace('\\', '-').replace(':', '-').replace('*', '-').replace('?', '-').replace('<', '-').replace('>', '-').replace('|', '-').replace('"', '-').strip('.')
        
        return title


def main():
    """
    Main function that orchestrates the entire process
    
    This is where the magic happens - it:
    1. Asks user for a video/playlist URL
    2. Determines if it's a single video or playlist
    3. Downloads and processes accordingly
    4. Creates beautiful PDFs with timestamps
    """
    print(" YouTube Video to PDF Frame Extractor Started!")
    print("=" * 50)
    
    # Get the YouTube URL from user
    url = input("📎 Enter the YouTube video or playlist URL: ")
    print(f" Processing URL: {url}")
    
    # Try to extract video ID to determine if it's a single video
    video_id = get_video_id(url)
    
    if video_id:  
        # It's a single video URL
        print(" Single video detected! Starting download...")
        
        try:
            # Download the video
            video_file = download_video(url, "video.mp4")
            if not video_file:
                print(" Failed to download video.")
                return
            
            print(" Video downloaded successfully!")
            print("  Getting video title...")
            
            # Get video title for PDF filename
            video_title = get_video_title(url)
            output_pdf_name = f"{video_title}.pdf"
            
            print(f" Creating PDF: {output_pdf_name}")
            print(" Extracting unique frames... (this may take a while)")
            
            # Create temporary folder for frames and process video
            with tempfile.TemporaryDirectory() as temp_folder:
                timestamps = extract_unique_frames(video_file, temp_folder)
                print(f" Found {len(timestamps)} unique frames!")
                print(" Converting frames to PDF...")
                convert_frames_to_pdf(temp_folder, output_pdf_name, timestamps)
            
            # Clean up downloaded video file
            os.remove(video_file)
            print(f" PDF created successfully: {output_pdf_name}")
            
        except Exception as e:
            print(f" Error processing video: {e}")
    
    elif "playlist" in url or "list=" in url:  
        # It's a playlist URL
        print(" Playlist detected! Getting video list...")
        
        try:
            # Get all videos in the playlist
            video_urls = get_playlist_videos(url)
            print(f"📹 Found {len(video_urls)} videos in playlist!")
            
            # Process each video in the playlist
            for i, video_url in enumerate(video_urls, 1):
                print(f"\n Processing video {i}/{len(video_urls)}")
                
                try:
                    # Download current video
                    video_file = download_video(video_url, "video.mp4")
                    if not video_file:
                        print(f" Skipping video {i} - download failed")
                        continue
                    
                    print(" Video downloaded!")
                    
                    # Get video title and create PDF
                    video_title = get_video_title(video_url)
                    output_pdf_name = f"{video_title}.pdf"
                    
                    print(f" Creating PDF: {output_pdf_name}")
                    print(" Extracting unique frames...")
                    
                    # Process frames and create PDF
                    with tempfile.TemporaryDirectory() as temp_folder:
                        timestamps = extract_unique_frames(video_file, temp_folder)
                        print(f"📸 Found {len(timestamps)} unique frames!")
                        convert_frames_to_pdf(temp_folder, output_pdf_name, timestamps)
                    
                    # Clean up
                    os.remove(video_file)
                    print(f"PDF created: {output_pdf_name}")
                    
                except Exception as e:
                    print(f" Error processing video {i}: {e}")
                    continue
            
            print(f"\n Playlist processing complete!")
            
        except Exception as e:
            print(f" Error processing playlist: {e}")
    
    else:
        # URL format not recognized
        print(" Invalid URL or unable to determine video/playlist type.")
        print(" Supported formats:")
        print("   • youtube.com/watch?v=VIDEO_ID")
        print("   • youtu.be/VIDEO_ID") 
        print("   • youtube.com/shorts/VIDEO_ID")
        print("   • youtube.com/playlist?list=PLAYLIST_ID")


# This is the entry point - run the main function when script is executed
if __name__ == "__main__":
    main()