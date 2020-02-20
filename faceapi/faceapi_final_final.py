import cv2
import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import glob
import requests
import json
import numpy as np
import os
import re
import asyncio
import io
import sys
import time
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType
indexx = 0
area_size = 0

# Set the FACE_SUBSCRIPTION_KEY from your Face service in Azure
KEY = os.environ['FACE_SUBSCRIPTION_KEY']

# Set the FACE_ENDPOINT environment variable with the endpoint from your Face service in Azure.
# This endpoint will be used in all examples in this quickstart.
ENDPOINT = os.environ['FACE_ENDPOINT'
                     ]
# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# Set the subscription_key from your Face service in Azure.
subscription_key = 'YOUR SUBSCRIPTION_KEY'
assert subscription_key

# Replace <My Endpoint String> with the string from your endpoint URL
face_api_url = 'https://<My Endpoint String>.com/face/v1.0/detect'

# Set the parameters for the result
headers = {'Ocp-Apim-Subscription-Key': subscription_key}
params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'age,gender',
}

# getfacefeature gets input of the local path of folder which image is located EX) "C:/Users/kkang/Documents/GitHub/faceapi/image"
def getfacefeature(local_path):
    try:
        print("Azure Blob storage v12 - Python quickstart sample")
        ## Set the AZURE_STORAGE_CONNECTION_STRING from your Azure storage service in Azure.
        connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        # Create the BlobServiceClient object which will be used to create a container client
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        
        # Create your container with a full public access
        # put your container name
        container_name = str("CONTAINER NAME")
       

        # Create a file in local Documents directory to upload and download
        #bring all jpg files in the folder
        images = glob.glob(local_path+'/*.jpg') 
        for i in images:
            number = re.sub('[^0-9]', '', i)
            number = int(number)
            local_file_name = str(i)
            print(i)

            upload_file_path = os.path.join(local_path, local_file_name)
            # Create a blob client using the local file name as the name for the blob
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)

            print("\nUploading to Azure Storage as blob:\n\t" + local_file_name)

    # Upload the created file to the azure server
            with open(upload_file_path, "rb") as data:
                blob_client.upload_blob(data)
                # my azure storage folder path, EX) 'https://qfqewfegqe.blob.core.windows.net/abcd/C:/Users/kkang/Documents/GitHub/faceapi/image/%d.jpg'%number
                # when my "STORAGE_NAME" =  qfqewfegqe, "CONTAINER_NAME" = abcd, 
                image_url = 'https://"STORAGE_NAME".blob.core.windows.net/"CONTAINER_NAME"/'+local_path+'/%d.jpg'%number  


               
                # Detect a face in an image that contains a single face
                single_image_name = os.path.basename(image_url)
                detected_faces = face_client.face.detect_with_url(url=image_url)
                if not detected_faces:
                    raise Exception('No face detected from image {}'.format(single_image_name))

                    # Convert width height to a point in a rectangle
                def getRectangle(faceDictionary):
                    rect = faceDictionary.face_rectangle
                    left = rect.left
                    top = rect.top
                    right = left + rect.width
                    bottom = top + rect.height

                    return ((left, top), (right, bottom))


                # Download the image from the url
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))

              
        
            '''
            when multiple images are detected, we compare the size of the boundary box around the face, and suppose that person who has
            biggest size of face is the frontmost person who is the real buyer at the counter
            '''
            for face in detected_faces:
                dam = getRectangle(face)
                area = (dam[0][0], dam[0][1], dam[1][0]+1, dam[1][1]+1)

                response = requests.post(face_api_url, params=params,
                                 headers=headers, json={"url": image_url})
                result = response.json()
                # measure the size of boundary box of each face detected in the image
                new_area_size = (dam[1][0]+1-dam[0][0]) * (dam[1][1]+1-dam[0][1])

                if new_area_size > area_size:
                    new_result = result
                else:
                    break
            # print the feature of the person EX) ['a4d59ffb-4d65-4300-ae60-dbf58d71d65b', 'male', 28.0]
            get_result= [new_result[0]['faceId'], new_result[0]['faceAttributes']['gender'],new_result[0]['faceAttributes']['age'] ]
            print(get_result)

    except Exception as ex:
        print('Exception:')
        print(ex)

 
getfacefeature("C:/Users/kkang/Documents/GitHub/faceapi/image")
