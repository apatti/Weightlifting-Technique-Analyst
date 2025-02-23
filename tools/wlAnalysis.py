import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import time 
import math
import json


def analyze_weightLifting_video( video_path: str,slow_factor, st) -> None:
    """
    Analyzes an Olympic weightlifting video for pose and provides basic analysis.

    Args:
        video_path (str): The path to the video file to analyze.


    Returns:
        None. Displays the video with the pose landmarks and print analysis
    """
    #object_detection_model = load_object_detection_model()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
  
# Get video properties  
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
      

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    frame_count = 0
    time_data = [] #store time data
    previous_hip_y = None
    previous_shoulder_y = None
    previous_wrist_y = None
    frame_count = 0
    analysis_results = []
    knee_angle_data = {"left": [], "right": []} #store knee angle data
    detection_status = {"first_phase": False, "extension": False, "transition": False, "catch": False}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            analysis, hip_y, shoulder_y, wrist_y = analyze_pose(
                mp_pose,results.pose_landmarks, 
                frame_width, frame_height,
                previous_hip_y, previous_shoulder_y, previous_wrist_y)
            time_data.append(time.time())
            knee_angle_data["left"].append(analysis.get("left_knee_angle", None))
            knee_angle_data["right"].append(analysis.get("right_knee_angle", None))
            
            if analysis.get("lift_phase") == "First Phase" and not detection_status["first_phase"]:
                detection_status["first_phase"] = True
            
            if analysis.get("lift_phase") == "Extension/Catch" and not detection_status["extension"]:
                detection_status["extension"] = True
                analysis["lift_phase"] = "Extension"
            
            if not detection_status["first_phase"]:
                analysis["lift_phase"] = "Setup"

            analysis_results.append({
                "frame": frame_count,
                "phase": analysis.get("lift_phase"),
                "bar_path_deviation": analysis.get("bar_path_deviation"),
                "hip_shoulder_movement": analysis.get("hip_shoulder_movement"),
                "left_knee_angle": analysis.get("left_knee_angle"),
                "right_knee_angle": analysis.get("right_knee_angle")
            })
            #print(f"Frame {frame_count}:", analysis) #print analysis for each frame.
            #detect_barbell(frame,object_detection_model)
            #detect_circle(frame)
            cv2.putText(frame, f"{analysis.get("lift_phase")}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            previous_hip_y = hip_y
            previous_shoulder_y = shoulder_y
            previous_wrist_y = wrist_y

        frame_count += 1
        #22.371817 250.71399   34.990044 261.4342
        #left, right, top, bottom = int(22.371817 * frame_width), int(250.71399 * frame_width), int(34.990044 * frame_height), int(261.4342 * frame_height)

        #cv2.rectangle(frame, (int(22.371817* frame_width), int(34.990044* frame_width)), (int(250.71399*frame_height), int(261.4342*frame_height)), (0, 255, 0), 2)
        if st:
            stFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(stFrame)
        else:
            cv2.imshow('Weightlifting Analysis', frame)
            if cv2.waitKey(int(1000 / 30 * slow_factor)) & 0xFF == ord('q'): # Slowed playback
                break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

    if not st:
        plot_knee_angles(time_data, knee_angle_data)
    #print(f"Analysis Results:{json.dumps(analysis_results)}")
    return analysis_results

def calculate_angle(a, b, c):
    a = np.array(a)  # First  
    b = np.array(b)  # Mid  
    c = np.array(c)  # End  
  
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])  
    angle = abs(radians * 180.0 / math.pi)  
  
    if angle > 180.0:  
        angle = 360 - angle  
  
    return angle

def analyze_pose(mp_pose, landmarks, frame_width, frame_height,previous_hip_y=None, previous_shoulder_y=None, previous_wrist_y=None):
    """
    Analyzes pose landmarks for basic feedback

    Args:
        mp_pose (mediapipe.solutions.pose): Mediapipe pose module.
        landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): Pose landmarks.
        image_shape (tuple): Shape of the image (height, width, channels).

    Returns:
        dict: Dictionary containing pose analysis data.
    """
    analysis = {}
    landmarks_list = landmarks.landmark

    def get_landmark(index):
        landmark = landmarks_list[index]
        return np.array([landmark.x * frame_width, landmark.y * frame_height])

    left_shoulder = get_landmark(mp_pose.PoseLandmark.LEFT_SHOULDER)
    right_shoulder = get_landmark(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    left_elbow = get_landmark(mp_pose.PoseLandmark.LEFT_ELBOW)
    right_elbow = get_landmark(mp_pose.PoseLandmark.RIGHT_ELBOW)
    left_wrist = get_landmark(mp_pose.PoseLandmark.LEFT_WRIST)
    right_wrist = get_landmark(mp_pose.PoseLandmark.RIGHT_WRIST)
    left_hip = get_landmark(mp_pose.PoseLandmark.LEFT_HIP)
    right_hip = get_landmark(mp_pose.PoseLandmark.RIGHT_HIP)
    left_knee = get_landmark(mp_pose.PoseLandmark.LEFT_KNEE)
    right_knee = get_landmark(mp_pose.PoseLandmark.RIGHT_KNEE)
    left_ankle = get_landmark(mp_pose.PoseLandmark.LEFT_ANKLE)
    right_ankle = get_landmark(mp_pose.PoseLandmark.RIGHT_ANKLE)

    shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
    analysis["shoulder_alignment"] = "Aligned" if shoulder_diff < 30 else "Misaligned"

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    
    if left_knee_angle < 100:  # Adjust thresholds as needed
        analysis["lift_phase"] =  "First Phase"
    elif left_knee_angle > 140:
        analysis["lift_phase"] = "Extension/Catch"
    else:
        analysis["lift_phase"] = "Transition"

    analysis["left_knee_angle"] = left_knee_angle
    analysis["right_knee_angle"] = right_knee_angle

    # Bar path analysis
    if previous_wrist_y is not None:
        if left_wrist[1] > previous_wrist_y + 10:  # Adjust threshold
            analysis["bar_path_deviation"] = "Bar path deviated forward at knee level."
        else:
            analysis["bar_path_deviation"] = None

    # Hip/Shoulder analysis
    if previous_hip_y is not None and previous_shoulder_y is not None:
        if left_hip[1] < previous_hip_y and left_shoulder[1] > previous_shoulder_y: #adjust thresholds
            analysis["hip_shoulder_movement"] = "Hips rose before shoulders."
        else:
            analysis["hip_shoulder_movement"] = None

    return analysis, left_hip[1], left_shoulder[1], left_wrist[1]
    
def plot_knee_angles(time_data, knee_angle_data):
    """Plots knee angles over time."""
    if not time_data or not knee_angle_data["left"] or not knee_angle_data["right"]:
        print("No data to plot.")
        return

    relative_time = [t - time_data[0] for t in time_data] #relative time

    plt.figure(figsize=(10, 6))
    plt.plot(relative_time, knee_angle_data["left"], label="Left Knee Angle")
    plt.plot(relative_time, knee_angle_data["right"], label="Right Knee Angle")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Knee Angle (degrees)")
    plt.title("Knee Angles Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    #path = input("Enter the path to the video file: ")
    analyze_weightLifting_video("clean.mp4",slow_factor=2)