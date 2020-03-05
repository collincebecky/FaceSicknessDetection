#PYTORCH

from   facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from   torch.utils.data import DataLoader
from   torchvision import datasets
import numpy as np
import pandas as pd
import os
import math
import face_alignment
from skimage import io
from sklearn import neighbors
import pickle

#from deep_face import FaceRecognition

# cuda for CUDA

###############################################################################

"use cuda if its available for my case I had cuda installed on my local environment .........................."

workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
#################################################################################
model_path= "trained_faces_model.clf"
classifier=None

# Loading model 
try:
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)

except Exception as e:
    print("[ERROR]:>>> Could not load Face Models! \n",e)




def train(model_save_path=model_path, n_neighbors=None, knn_algo='ball_tree', verbose=False):

   

    # Loop through each person in the training set
    def collate_fn(x):return x[0]

    dataset = datasets.ImageFolder("./DataSet")
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    All_Embeddings=[]

    labels = []
    aligned = []
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)
            labels.append(dataset.idx_to_class[y])

    for data in aligned:
    	
    	aligned = torch.stack([data]).to(device)

    	embeddings = resnet(aligned).cpu().detach().numpy().flatten()
    	All_Embeddings.append(embeddings)

    print("CREATING EMBEDDINGS ........................................")

                   
    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:

        n_neighbors = int(round(math.sqrt(len(All_Embeddings))))
        print("Chose n_neighbors automatically:", n_neighbors)
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    classifier = neighbors.KNeighborsClassifier(
        n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    classifier.fit(All_Embeddings, labels)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(classifier, f)

    return classifier
 

if __name__ == '__main__':
	train()
