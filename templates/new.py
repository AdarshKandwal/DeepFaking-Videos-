import cv2  
  
# Load the cascade  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
  
# To capture video from existing video.   
cap = cv2.VideoCapture('00.mp4')  
count=0 
while True:  
    # Read the frame  
    _, img = cap.read()  
    # Convert to grayscale  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    file_name_path='user'+str(count)+'.jpg'
    
    # Detect the faces  
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  
    # Draw the rectangle around each face  
    for (x, y, w, h) in faces:  
        face = img[y:y+h, x:x+w] 
        cv2.imwrite(str(count)+'.jpg', face) 
        count+=1
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  
    # Display  
    cv2.imshow('Video', img)  
    cv2.imwrite(file_name_path,faces)

    # Stop if escape key is pressed  
    k = cv2.waitKey(30) & 0xff  
    if k==27:  
        break           
# Release the VideoCapture object  
cap.release() 