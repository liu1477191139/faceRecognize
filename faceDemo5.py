import face_recognition
import cv2
import numpy as np

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
long_image = face_recognition.load_image_file("imgs/1 (2).jpg")
long_face_encoding = face_recognition.face_encodings(long_image)[0]

# Load a second sample picture and learn how to recognize it.
xiaokkk_image = face_recognition.load_image_file("imgs/2 (1).jpg")
xiaokkk_face_encoding = face_recognition.face_encodings(xiaokkk_image)[0]

# Load a third sample picture and learn how to recognize it.
heng_image = face_recognition.load_image_file("imgs/3 (2).jpg")
heng_face_encoding = face_recognition.face_encodings(heng_image)[0]

# Load a other sample picture and learn how to recognize it.
dz_image = face_recognition.load_image_file("imgs/党政~1.jpg")
dz_face_encoding = face_recognition.face_encodings(dz_image)[0]

pf_image = face_recognition.load_image_file("imgs/彭芳~1.jpg")
pf_face_encoding = face_recognition.face_encodings(pf_image)[0]

lz_image = face_recognition.load_image_file("imgs/梁钟~1.jpg")
lz_face_encoding = face_recognition.face_encodings(lz_image)[0]

xj_image = face_recognition.load_image_file("imgs/熊杰~1.jpg")
xj_face_encoding = face_recognition.face_encodings(xj_image)[0]

jjw_image = face_recognition.load_image_file("imgs/荆嘉伟~1.jpg")
jjw_face_encoding = face_recognition.face_encodings(jjw_image)[0]

jzy_image = face_recognition.load_image_file("imgs/贾紫英~1.jpg")
jzy_face_encoding = face_recognition.face_encodings(jzy_image)[0]

zzh_image = face_recognition.load_image_file("imgs/赵子豪~1.jpg")
zzh_face_encoding = face_recognition.face_encodings(zzh_image)[0]

cjq_image = face_recognition.load_image_file("imgs/陈嘉祺~1.jpg")
cjq_face_encoding = face_recognition.face_encodings(cjq_image)[0]

mjp_image = face_recognition.load_image_file("imgs/马菁鹏~1.jpg")
mjp_face_encoding = face_recognition.face_encodings(mjp_image)[0]



# Create arrays of known face encodings and their names
known_face_encodings = [
    long_face_encoding,
    xiaokkk_face_encoding,
    heng_face_encoding,
    dz_face_encoding,
    pf_face_encoding,
    lz_face_encoding,
    xj_face_encoding,
    jjw_face_encoding,
    jzy_face_encoding,
    zzh_face_encoding,
    cjq_face_encoding,
    mjp_face_encoding
]






known_face_names = [
    "long",
    "xiaokkk",
    "heng",
    "zheng",
    "F",
    "zhong",
    "jie",
    "wei",
    "ying",
    "hao",
    "Q",
    "peng"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
