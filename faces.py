import face_recognition
from PIL import Image, ImageDraw
import numpy as np

obama_image = face_recognition.load_image_file("known_people/barack_obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

bradley_image = face_recognition.load_image_file("known_people/bradley_cooper.jpg")
bradley_face_encoding = face_recognition.face_encodings(bradley_image)[0]

known_face_encodings = [
    obama_face_encoding,
    bradley_face_encoding
]

known_face_names = [
    "obama",
    "bradley"
]

unknown_image = face_recognition.load_image_file("unknown_people/unknown2.jpg")

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image)
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_matches_index = np.argmin(face_distances)
    if matches[best_matches_index]:
        name = known_face_names[best_matches_index]
        
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 2, bottom - text_height - 2), name, fill=(255, 255, 255, 255))

del draw
pil_image.show()
