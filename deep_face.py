from face_recognition.face_recognition_cli import image_files_in_folder
from PIL import Image, ImageDraw
from sklearn import neighbors
import face_recognition
import cv2, imutils
import random as r
import os.path
import pickle
import math
import os

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
model_path= "trained_faces_model.clf"
classifier=None

# Loading model 
try:
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)

except Exception as e:
    print("[ERROR]:>>> Could not load Face Models! \n",e)

class FaceRecognition(object):
    """docstring for FaceRecognition"""

    def __init__(self, arg=None):
        super(FaceRecognition, self).__init__()
        self.arg = arg
        self.people = {'unknown':(0,0,245)}
    

    def train(self, train_dir, model_save_path=model_path, n_neighbors=None, knn_algo='ball_tree', verbose=False):
      
        train = []
        test  = []

        # Loop through each person in the training set
        for class_dir in os.listdir(train_dir):
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue

            face_count = 0
            # Loop through each training image for the current person
            for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)
                print("[Training]: checking ",img_path)
                if len(face_bounding_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                            face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    train.append(face_recognition.face_encodings(
                        image, known_face_locations=face_bounding_boxes)[0])
                    test.append(class_dir)
                    face_count += 1
                    if face_count == 25:
                       break


        # Determine how many neighbors to use for weighting in the KNN classifier
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(train))))
            print("Chose n_neighbors automatically:", n_neighbors)
            if verbose:
                print("Chose n_neighbors automatically:", n_neighbors)

        # Create and train the KNN classifier
        classifier = neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        classifier.fit(train, test)

        # Save the trained KNN classifier
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(classifier, f)

        return classifier
 

    
    def predict(self, img=None, distance_threshold=0.45, zombie=False):
        if img is None:
            return
        img = imutils.resize(img, width=400)
        face_locations = face_recognition.face_locations(img)

        if len(face_locations) == 0 :
            #print("Face recg predictions: No face")
            return None 
         
        faces_encodings = face_recognition.face_encodings(img, known_face_locations=face_locations)
        closest_match = classifier.kneighbors(faces_encodings, n_neighbors=3)
        are_matches   = [closest_match[0][i][0] <=
                       distance_threshold for i in range(len(face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(
            classifier.predict(faces_encodings), face_locations, are_matches)]
        

        if zombie:
            #return label, confidence;
            return predictions
                
        return predictions


    def predict_from_file(self, img_path="face_lab/train", distance_threshold=0.5):
 
        if img_path is None:
            return
        
        if not os.path.isfile(img_path) or os.path.splitext(img_path)[1][1:] not in ALLOWED_EXTENSIONS:
            raise Exception("Invalid image path: {}".format(img_path))
        else:
            #Load image file
            img = face_recognition.load_image_file(img_path)

         
        #Resize
        img = imutils.resize(img, width=400)

        # find face locations
        face_locations = face_recognition.face_locations(img)

        # If no faces are found in the image, return None.
        if len(face_locations) == 0:
            #print("Face recg predictions: No face")
            return None 
         
        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(img, known_face_locations=face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_match = classifier.kneighbors(faces_encodings, n_neighbors=3)
        are_matches   = [closest_match[0][i][0] <=
                       distance_threshold for i in range(len(face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(
            classifier.predict(faces_encodings), face_locations, are_matches)]
        
        return predictions



    def label(self, img=None, predictions=None, path=None, preview=False):
        if img is None and path is None:
            return

        if path is not None:
            img = face_recognition.load_image_file(path)

        img = imutils.resize(img, width=400)

        #top, right, bottom, left = predictions
        for name, (top, right, bottom, left) in (predictions):
            faceImage = img[top-35:bottom+35 , left-35:right+35]
            name.lower()
            try:
                color = self.people[name.lower()]
            except:
                color = (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255))
                self.people[name.lower()] = color
            font = cv2.FONT_HERSHEY_DUPLEX

            try:
                Id, name, info = name.split('..')
            except Exception as e:
                Id = info = name 

            #[y:y+h, x:x+w]




            cv2.rectangle(img, (left, top), (right, bottom+25), color, 1)
            # Draw a label with a name below the face
            cv2.rectangle(img, (left, bottom + 25), (right, bottom), color, cv2.FILLED)
            try:
                cv2.putText(img, name.split('-')[0], (left+5, bottom + 20),font, 0.5, (255, 255, 255), 1)
            except:
                cv2.putText(img, name , (left+5, bottom + 20),font, 0.5, (255, 255, 255), 1)
           
                                                 
        if preview:
            pil_image = Image.fromarray(img)
            pil_image.show()

        return (img , Id, name, info,faceImage)

      

def main(data=None):
    fr = FaceRecognition()
    fr.people['unknown']= (245,0,0)

        
    if data is not None:
        src = cv2.VideoCapture(data)
        while src.isOpened():
            true,img=src.read()
            if true:
                predictions=fr.predict(img=img)
                if predictions is None:
                    continue
                print(predictions)
                fr.label(img=img,predictions=predictions, preview=True)
            if KeyboardInterrupt:
                break
        return
    """Training  classifier """
    global classifier
    classifier = fr.train("face_lab/train", n_neighbors=5)
    print("Training complete!")
    
    """Testining  classifier """
    for image_file in os.listdir("face_lab/test"):
        full_file_path = os.path.join("face_lab/test", image_file)
        print("Loading faces from:  {}".format(image_file))
        
        predictions = fr.predict_from_file(full_file_path)

        # Show results
        if predictions is not None:
            for name, (top, right, bottom, left) in predictions:
                print("- Found {} at ({}, {})".format(name, left, top))

            fr.label(path=full_file_path,predictions=predictions,preview=True)

if __name__ == "__main__":
    main()
    
