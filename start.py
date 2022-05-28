import io
from PIL import Image
import streamlit as st
import numpy as np
import torch
from torchvision import transforms

MODEL_PATH = 'useful_files/model_DenseNet.pt'
LABELS_PATH = 'model_classes.txt'


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def load_model(model_path):
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    return model


def load_labels(labels_file):
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories


def predict1(model, categories, image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    all_prob, all_catid = torch.topk(probabilities, len(categories))
    for i in range(all_prob.size(0)):
        st.write(categories[all_catid[i]], all_prob[i].item())
        
        
def process_image(image):
    #img = Image.open(image)
    img = image
    ##########Scales 
    if img.size[0] > img.size[1]:
        img.thumbnail((1000000, 256))
    else:
        img.thumbnail((256 ,1000000))
    #######Crops: to crop the image we have to specifiy the left,Right,button and the top pixels because the crop function take a rectongle ot pixels
    Left = (img.width - 224) / 2
    Right = Left + 224
    Top = (img.height - 244) / 2
    Buttom = Top + 224
    img = img.crop((Left, Top, Right, Buttom))
    img = np.stack((img,)*3, axis=-1)# to repeate the the one chanel of a gray image to be RGB image 
    #img = np.repeat(image[..., np.newaxis], 3, -1)
    #print(np.array(img).shape)
    #normalization (divide the image by 255 so the value of the channels will be between 0 and 1 and substract the mean and divide the result by the standtared deviation)
    img = ((np.array(img) / 255) - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = img.transpose((2, 0, 1))
    return img    
 


def predict(image_path, model, topk=2):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_image(image_path)
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    model_input = image_tensor.unsqueeze(0)
    probs = torch.exp(model.forward(model_input))
    
    
    # Top probs
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    #top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[lab] for lab in top_labs]
    st.write(top_probs, top_flowers)
    
    #for i in range(all_prob.size(0)):
        #st.write(categories[all_catid[i]], all_prob[i].item())
    #return top_probs, top_flowers



def main():
    st.title('Custom model demo')
    model = load_model(MODEL_PATH)
    categories = load_labels(LABELS_PATH)
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        #predict(model, categories, image)
        predict( image, model, 2)


if __name__ == '__main__':
    main()
