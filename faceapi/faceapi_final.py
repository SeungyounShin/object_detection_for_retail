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
fcount = 0
mcount = 0
command = None
KEY = os.environ['FACE_SUBSCRIPTION_KEY']

# Set the FACE_ENDPOINT environment variable with the endpoint from your Face service in Azure.
# This endpoint will be used in all examples in this quickstart.
ENDPOINT = os.environ['FACE_ENDPOINT']
# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# set to your own subscription key value
subscription_key = 'ad8433203b0a44aa803094e2eb780b21'
assert subscription_key

# replace <My Endpoint String> with the string from your endpoint URL
face_api_url = 'https://kpmgfaceapi.cognitiveservices.azure.com/face/v1.0/detect'


vid = cv2.VideoCapture('C:/Users/kkang/Documents/GitHub/faceapi/test.mp4')
#for frame identity
index = 0
while(True):
    # Extract images
    ret, frame = vid.read()
    # end of frames
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    cv2.imshow('frame', rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Saves images
        name = 'C:/Users/kkang/Documents/GitHub/faceapi/images/' + str(index) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        break
vid.release()
cv2.destroyAllWindows()

try:
    print("Azure Blob storage v12 - Python quickstart sample")
    # Quick start code goes here
    # Retrieve the connection string for use with the application. The storage
    # connection string is stored in an environment variable on the machine
    # running the application called AZURE_STORAGE_CONNECTION_STRING. If the environment variable is
    # created after the application is launched in a console or with Visual Studio,
    # the shell or application needs to be closed and reloaded to take the
    # environment variable into account.
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Create a unique name for the container
    container_name = str("abcd")
    # Create the container
    #container_client = blob_service_client.create_container(container_name,)

    # Create a file in local Documents directory to upload and download
    local_path = "C:/Users/kkang/Documents/GitHub/faceapi/images"  #use your local path
    images = glob.glob(local_path+'/*.jpg') #bring all jpg files in the folder
    for i in images:
        number = re.sub('[^0-9]', '', i)
        number = int(number)
        local_file_name = str(i)
        upload_file_path = os.path.join(local_path, local_file_name)
        # Create a blob client using the local file name as the name for the blob
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)

        print("\nUploading to Azure Storage as blob:\n\t" + local_file_name)

    # Upload the created file
        with open(upload_file_path, "rb") as data:
            blob_client.upload_blob(data)

        image_url = 'https://qfqewfegqe.blob.core.windows.net/abcd/C:/Users/kkang/Documents/GitHub/faceapi/images/%d.jpg'%number  # my azure storage folder path

        indexx = 0
        # Set the FACE_SUBSCRIPTION_KEY environment variable with your key as the value.
        # This key will serve all examples in this document.
        # Detect a face in an image that contains a single face
        single_face_image_url = image_url
        single_image_name = os.path.basename(single_face_image_url)
        detected_faces = face_client.face.detect_with_url(url=single_face_image_url)
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
        response = requests.get(single_face_image_url)
        img = Image.open(BytesIO(response.content))

        # For each face returned use the face rectangle and draw a red box.
        print('Drawing rectangle around face... see popup for results.')
        draw = ImageDraw.Draw(img)
        for face in detected_faces:
            draw.rectangle(getRectangle(face), outline='red')
            dam = getRectangle(face)
            area = (dam[0][0], dam[0][1], dam[1][0]+1, dam[1][1]+1)
            print(area)
            img.show()
            crop_img = img.crop(area)
            crop_img.show()
            crop_img.save("C:/Users/kkang/Documents/GitHub/faceapi/crop_img/crop%d.jpg"%index)
            indexx+=1

        headers = {'Ocp-Apim-Subscription-Key': subscription_key}

        params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise'
        }
        #'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise'
        response = requests.post(face_api_url, params=params,
                    headers=headers, json={"url": image_url})
        ram = response.json()
        print(ram)

        # judge gender based on many frames
        genderjudge = ram[0]['faceAttributes']['gender']
        agejudge = ram[0]['faceAttributes']['age']
        print("Person's age is", agejudge)
        if genderjudge == "female":
            fcount+=1
        else:
            mcount+=1

    if fcount > mcount:
        print("Person's gender is female")
    else:
        print("Person's gender is male")


except Exception as ex:
    print('Exception:')
    print(ex)
