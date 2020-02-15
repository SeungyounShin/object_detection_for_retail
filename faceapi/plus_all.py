import cv2
import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import glob
import requests
import json
import numpy as np
import os
import re
fcount = 0
mcount = 0
# set to your own subscription key value
subscription_key = "your face api key"
assert subscription_key

# replace <My Endpoint String> with the string from your endpoint URL
face_api_url =  <My Endpoint String>


vid = cv2.VideoCapture('C:/Users/kkang/Documents/GitHub/faceapi/test.mp4')
#for frame identity
index = 0
while(True):
    # Extract images
    ret, frame = vid.read()
    # end of frames
    if not ret:
        break
    # Saves images
    name = 'C:/Users/kkang/Documents/GitHub/faceapi/images/' + str(index) + '.jpg'
    print ('Creating...' + name)
    cv2.imwrite(name, frame)

    # next frame
    index += 100
    if index >5000:
        break


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
