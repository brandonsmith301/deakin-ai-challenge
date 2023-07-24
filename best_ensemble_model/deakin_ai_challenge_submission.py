# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2023. Reda Bouadjenek, Deakin University                      +
#     Email:  reda.bouadjenek@deakin.edu.au                                    +
#                                                                              +
#  Licensed under the Apache License, Version 2.0 (the "License");             +
#   you may not use this file except in compliance with the License.           +
#    You may obtain a copy of the License at:                                  +
#                                                                              +
#                 http://www.apache.org/licenses/LICENSE-2.0                   +
#                                                                              +
#    Unless required by applicable law or agreed to in writing, software       +
#    distributed under the License is distributed on an "AS IS" BASIS,         +
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  +
#    See the License for the specific language governing permissions and       +
#    limitations under the License.                                            +
#                                                                              +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import sys, os, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViltProcessor
from transformers import AutoProcessor, BlipForQuestionAnswering
from transformers.tokenization_utils_base import BatchEncoding
from PIL import Image
import json
import torch
import wandb

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, images, questions, processor):
        self.images = images
        self.questions = questions
        self.processor = processor   

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = img.resize((256, 256))
        encoding = self.processor(img, self.questions[idx], padding="max_length", truncation=True, return_tensors="pt")
        item = {key: val.squeeze() for key, val in encoding.items()}
        return item
    
if __name__ == "__main__":
    if len(sys.argv) == 1:
        input_dir = '.'
        output_dir = '.'
    else:
        input_dir = os.path.abspath(sys.argv[1])
        output_dir = os.path.abspath(sys.argv[2])

    print("Using input_dir: " + input_dir)
    print("Using output_dir: " + output_dir)
    print(sys.version)

    # Download from weights from wandb
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)
    run = wandb.init()
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load Model 1
    path_1 = "" # add your path here from wandb (pre-trained weights)
    processor_1 = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa") 
    path_1 = wandb.use_artifact(path_1).download()
    model_1 = torch.load(path_1 + "/model.pth", map_location=torch.device('cpu'))

    # Load Model 2
    path_2 = "" # add your path here from wandb (pre-trained weights)
    processor_2 = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
    path_2 = wandb.use_artifact(path_2).download()
    model_2 = torch.load(path_2 + "/model.pth", map_location=torch.device('cpu'))

    # Read test dataset
    imgs_path_test = input_dir + '/simpsons_test/'
    q_val_file = imgs_path_test + 'questions.json'
    q_test = json.load(open(q_val_file))

    def preprocessing(questions, imgs_path):
        # Make sure the questions and annotations are alligned
        questions['questions'] = sorted(questions['questions'], key=lambda x: x['question_id'])
        q_out = []
        imgs_out = []
        q_ids = []
        # Preprocess questions
        for q in questions['questions']:
            # Preprocessing the question
            q_text = q['question'].lower()
            q_out.append(q_text)
            file_name = imgs_path + str(q['image_id']) + '.png'
            imgs_out.append(file_name)
            q_ids.append(q['question_id'])
        return imgs_out, q_out, q_ids

    imgs_test, q_test, q_ids_test = preprocessing(q_test, imgs_path_test)
        
    # Define threshold for Model 1
    custom_threshold = 0.8149722814559937  

    # The models in evaluation mode
    model_1.eval()
    model_2.eval()

    # Prediction arrays
    y_predict1 = []
    y_predict2 = []

    # Dataloader for model 1
    dataset1 = VQADataset(imgs_test, q_test, processor_1)
    dataloader_1 = DataLoader(dataset1, batch_size=1)

    with torch.no_grad():
        for local_batch in dataloader_1:
            local_batch = BatchEncoding(local_batch)
            local_batch.to(device)
            outputs1 = model_1(**local_batch)
            probabilities = torch.nn.functional.softmax(outputs1.logits, dim=-1)
            pred1 = (probabilities[:, 1] > custom_threshold).cpu().numpy()
            y_predict1 += pred1.tolist()

    # Convert predictions to numpy array
    y_predict1 = np.array(y_predict1)

    # Prepare the dataloader for model 2
    dataset2 = VQADataset(imgs_test, q_test, processor_2)
    dataloader_2 = DataLoader(dataset2, batch_size=1)

    with torch.no_grad():
        for local_batch in dataloader_2:
            local_batch = BatchEncoding(local_batch)
            local_batch.to(device)
            outputs2 = model_2.generate(**local_batch)
            pred2 = processor_2.decode(outputs2[0], skip_special_tokens=True)
            y_predict2.append(1 if pred2 == 'yes' else 0)

    y_predict2 = np.array(y_predict2)
    y_predict_ensemble = np.logical_or(y_predict1, y_predict2)
 
    # Writting predictions to file.
    answers = ['no','yes']
    
    with open(os.path.join(output_dir, 'answers.txt'), 'w') as result_file:
        for i in range(len(y_predict_ensemble)):
            result_file.write(str(q_ids_test[i]) + ',' + answers[y_predict_ensemble[i]] + '\n')
