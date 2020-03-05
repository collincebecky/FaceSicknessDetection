from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, CreateView
from django.core.files.storage import FileSystemStorage
from django.urls import reverse_lazy

from .forms import BookForm
from .models import Book

#PYTORCH

from   facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from   torch.utils.data import DataLoader
from   torchvision import datasets
import numpy as np
import pandas as pd
import os

import face_alignment
from skimage import io
from pathlib import Path
import pickle

# cuda for CUDA

###############################################################################

"use cuda if its available for my case I had cuda installed on my local environment .........................."

workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
#################################################################################
#t######## Face alignment predictor
face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,face_detector='sfd',device='cpu')

classifier=None
model_path= "/home/brance/Mine/Prototypes/Face Ui/trained_faces_model.clf"
    
# Loading model 
try:
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)
        print("Loaded classifier ...................")

except Exception as e:
    print("[ERROR]:>>> Could Not Load FaceSickNess Classifier ! \n",e)    




class Home(TemplateView):
    template_name = 'home.html'


def run(request):
    

    context = {}
    print("GETTING HERE ...............")
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        save_file_name=""
        print("Name ....",uploaded_file.name)
        fs = FileSystemStorage()

        if Path('/home/brance/Mine/Prototypes/Face Ui/media/candidate/output.jpeg').is_file():
            print ("File exist")
            os.remove("/home/brance/Mine/Prototypes/Face Ui/media/candidate/output.jpeg")
        name = fs.save("output.jpeg", uploaded_file)
        context['url'] = fs.url(name)
        print("")
        print ("PATIENT >>>>>>>>>>>>>>>>>",diagnose_image())

        context["status"]= diagnose_image()[0]
	
        
       
    return render(request, 'home.html', context)

def diagnose_image():

    
    """
    input = io.imread('/home/brance/Mine/Prototypes/Face Ui/media')
    preds = fa.get_landmarks(input)

    print("LANDMARKS ....",preds)
    print("We should be seeing Landmarks ..........................")
    """

           
            #pass

    #out_image.save("../../media/output.jpeg")
    def collate_fn(x):return x[0]

    dataset = datasets.ImageFolder("../Face Ui/media/")
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    aligned = []
    names = []
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])

    
    aligned = torch.stack(aligned).to(device)

    embeddings = resnet(aligned).cpu().detach().numpy()
   # tensor_array.cpu().detach()

       # print("Locations ",face_locations)

        
       # cv2.rectangle(image, (startX, startY), (endX, endY),
           # (0, 0, 255), 2)

    pred = classifier.predict(embeddings)
    print("PREDICT PROBA >>>>>>>>>>>>>>>>",classifier.predict_proba(embeddings))
    return  pred

        
    
       


       # dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
       # print(pd.DataFrame(dists, columns=names, index=names))
            #draw = ImageDraw.Draw(out_image)
            #draw.rectangle(((left,top),(right,bottom)))
        #cv.imshow("window",image)

        #cv.imwrite("result.jpg",image)
         

    

        #for (top, right, bottom, left) in face_locations:
         #   out_image=out_image.crop((left-(out_image_w*.1), top-(out_image_h*.17), right+(out_image_w*.1), bottom+(out_image_h*.2)))


    



