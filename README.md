# Crowd-Gathering-Detection
Crowd Gathering Detection using YOLOv8 Detects individuals in video feeds, monitors line crossings, and flags gatherings based on proximity. Outputs annotated videos with bounding boxes, crowd counts, and alerts. Ideal for crowd management, security, and public safety. Built with YOLOv8, OpenCV, and Python.
Here’s a detailed description tailored for a README file:

---

# **Crowd Gathering Detection using YOLOv8**  

This project implements a robust system for detecting and analyzing human gatherings in video feeds using the YOLOv8 object detection model. It provides insights into crowd movement and density by identifying individuals, counting their interactions, and detecting gatherings based on their proximity. The system outputs annotated videos that visualize these detections, making it ideal for applications in public safety, event management, and security monitoring.  

---

## **Features**  

1. **Human Detection**  
   - Uses YOLOv8, a state-of-the-art object detection model, to identify individuals in video frames.  
   - Draws bounding boxes around detected people with confidence scores for accuracy assessment.  

2. **Line Crossing Counter**  
   - A virtual counting line is drawn on the video feed, and the system tracks individuals crossing this line.  
   - Helps monitor crowd flow by dynamically updating the count of people crossing.  

3. **Gathering Detection**  
   - Identifies gatherings by calculating the distance between detected individuals.  
   - Flags groups of people closer than a predefined threshold and highlights these groups with circles, lines, and alert labels.  

4. **Video Annotation**  
   - Outputs processed video with real-time annotations:  
     - Bounding boxes for detected individuals.  
     - Dynamic counters for total people in frame and line crossings.  
     - Alerts for detected gatherings.  

5. **Customizable Parameters**  
   - Threshold distances for gathering detection.  
   - Position and sensitivity of the counting line.  
   - Adjustable frame processing settings to adapt to different scenarios.  

---

## **Applications**  

- **Crowd Monitoring**: Assess densities in crowded public spaces such as malls, airports, and train stations.  
- **Security Surveillance**: Monitor restricted zones and detect unusual crowd behavior.  
- **Event Management**: Track attendee movement and ensure safety compliance during large gatherings.  
- **Public Safety**: Prevent overcrowding in sensitive areas by identifying risky gatherings early.  

---

## **Technologies Used**  

- **YOLOv8**: Pre-trained object detection model for identifying people in video feeds.  
- **OpenCV**: For video processing, visualization, and annotations.  
- **Python**: Programming language used for implementation and integration.  

---

## **Setup and Usage**  

1. **Install Dependencies**  
   Install the required libraries using pip:  
   ```bash
   pip install ultralytics opencv-python-headless numpy
   ```

2. **Run the Script**  
   Ensure the YOLO model file (`yolov8n.pt`) is downloaded. Place your video in the working directory and update the `video_path` variable in the script.  
   Execute the script:  
   ```bash
   python crowd_gathering.py
   ```

3. **Output**  
   - The processed video will be saved to the specified `output_path`.  
   - Annotations include detected individuals, gathering alerts, and line-crossing counts.  

---

## Customization  

- Adjust Gathering Detection: Modify the `threshold` parameter in the `detect_gatherings()` function to set the distance criteria for detecting gatherings.  
- Change Line Position: Update the `count_line_position` variable to reposition the counting line.  
- **Improve Detection Accuracy**: Replace `yolov8n.pt` with `yolov8s.pt` or `yolov8m.pt` for better precision.  

---

## Future Enhancements  

- Integrate live video feed capabilities for real-time monitoring.  
- Implement advanced crowd behavior analytics to predict potential risks.  
- Add alerts and notifications for detected gatherings.  

---

## License  
This project is licensed under the MIT License. Feel free to use and modify it as needed.  

---
