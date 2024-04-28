import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
from pathlib import Path
from gtts import gTTS
import random



# Load the model
model = YOLO('best.pt')


def predict_item(image_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    filename = Path(image_path).stem
    print(filename)

    results = model(image_path, stream=True)

    for i, result in enumerate(results):
        image_filename = f'{output_folder}/{filename}.jpg'
        labels_filename = f'{output_folder}/{filename}.txt'
        print(labels_filename)

        result.save(filename=image_filename)

        if hasattr(result, 'boxes'):
            boxes = result.boxes
            with open(labels_filename, 'w') as f:
                for box in boxes.data:
                    class_id = int(box[5]) 
                    label = result.names[class_id]
                    bbox = tuple(box[:4].cpu().numpy()) 
                    confidence = box[4].item() 
                    f.write(f"{label} {bbox} {confidence:.2f}\n") 

    def split_text_line(line):
        line = line.strip()

        if 'gm' in line:
            unit = 'gm'
        elif 'ml' in line:
            unit = 'ml'
        else:
            return None 
        
        quantity_end_index = line.find(unit) + 2 
        product_name = line[:quantity_end_index].rsplit(' ', 1)[0]
        quantity = line[quantity_end_index-4:quantity_end_index].strip()
        rest = line[quantity_end_index+1:].strip()

        return product_name, quantity, rest

    file_path = labels_filename

    with open(file_path, 'r') as file:
        for line in file:
            product_name, quantity, rest = split_text_line(line)
            print("Product Name:", product_name)
            print("Quantity:", quantity)
            print("The Rest:", rest) 
            return product_name


def main():
    st.set_page_config(page_title="VisionAI",
                   page_icon = '👓👓',
                   layout = 'centered',
                   initial_sidebar_state = 'collapsed')
    st.title("VISIONAI")

    st.image('a-blind-man-near-the-grocery-store.jpeg',width=500)

    st.markdown('''<div style="text-align: justify;">
                    See the Unseen: Empowering vision, Enhancing individual lives with AI.
                </div>''', unsafe_allow_html=True)

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image_path = uploaded_image.name
        print(image_path)
        image = Image.open(uploaded_image)

        output_folder = 'output'

        if st.button('Predict'):
            with st.spinner('Predicting...'):
                predicted_class = predict_item(image_path, output_folder)
                st.success(f"Prediction: {predicted_class}")
                if predicted_class:
                    # List of variations
                    variations = [
                        f"You possess the {predicted_class} product in your palm.",
                        f"You\'re holding onto the {predicted_class} product.",
                        f"You\'re in possession of the {predicted_class} product.",
                        f"You\'re carrying the {predicted_class} product in your hand.",
                        f"The {predicted_class} product is held firmly in your hand.",
                        f"You\'re gripping the {predicted_class} product.",
                        f"You\'re clutching the {predicted_class} product in your hand.",
                        f"The {predicted_class} product is in your possession.",
                        f"You have the {predicted_class} product in hand.",
                        f"You've got a firm grip on the {predicted_class} product.",
                        f"You're cradling the {predicted_class} product in your palm.",
                        f"The {predicted_class} product is clasped in your hand.",
                        f"You're carrying the {predicted_class} product with ease." 
                    ]

                    phrases = random.choice(variations)
                    print(phrases)
                    convert_to_speech(phrases)

def convert_to_speech(text):
    tts = gTTS(text,lang='en')
    tts.save('hello.mp3')
    os.system('afplay hello.mp3')            

if __name__ == "__main__":
    main()
