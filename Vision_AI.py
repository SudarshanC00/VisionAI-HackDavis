import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
from pathlib import Path
from gtts import gTTS
import random



# Load your trained ML model here
# For example, if you have a model saved as model.h5
# model = tf.keras.models.load_model('model.h5')

# Load the model
model = YOLO('/Users/sudarshanc/best.pt')


# Define the classes or labels for your predictions
# classes = ["class1", "class2", "class3"]  # Replace with your actual classes

def predict_item(image_path, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get the filename without extension
    filename = Path(image_path).stem
    print(filename)

    # Perform prediction
    results = model(image_path, stream=True)

    # Iterate over each result (assuming there's only one result per image for simplicity)
    for i, result in enumerate(results):
        # Filename for the detected image and labels text file
        image_filename = f'{output_folder}/{filename}.jpg'
        labels_filename = f'{output_folder}/{filename}.txt'
        print(labels_filename)

        # Save the detected image
        result.save(filename=image_filename)

        # Extract and save labels
        if hasattr(result, 'boxes'):
            boxes = result.boxes
            with open(labels_filename, 'w') as f:
                for box in boxes.data:
                    class_id = int(box[5])  # Access the class index
                    label = result.names[class_id]  # Retrieve the label using the class index
                    bbox = tuple(box[:4].cpu().numpy())  # Get bounding box coordinates
                    confidence = box[4].item()  # Get the confidence score
                    f.write(f"{label} {bbox} {confidence:.2f}\n")  # Save label, bbox, and confidence

    # with open(labels_filename, 'r') as file:
    #     for line in file:
    #         return line.strip()
        
    def split_text_line(line):
        line = line.strip()

        # Find the index for 'gm' or 'ml'
        if 'gm' in line:
            unit = 'gm'
        elif 'ml' in line:
            unit = 'ml'
        else:
            return None  # No known unit found
        
        # Find the end of the quantity which includes 'gm'
        quantity_end_index = line.find(unit) + 2  # +2 to include 'gm' in the quantity

        # Product name is everything before the quantity minus the space
        product_name = line[:quantity_end_index].rsplit(' ', 1)[0]

        # Quantity is the last part of the string before the coordinates
        quantity = line[quantity_end_index-4:quantity_end_index].strip()

        # The rest is everything after the quantity
        rest = line[quantity_end_index+1:].strip()

        return product_name, quantity, rest

    # Replace 'path_to_your_file.txt' with the path to your text file
    file_path = labels_filename

    # Open the file and process each line
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

    st.image('/Users/sudarshanc/a-blind-man-near-the-grocery-store.jpeg',width=500)

    st.markdown('''<div style="text-align: justify;">
                    Our application empowers blind individuals during grocery shopping 
        by providing real-time auditory descriptions of scanned products.
        Seamlessly integrating image recognition technology, it swiftly identifies
            items held by the user and articulates comprehensive details aloud, ensuring 
            an independent and informed shopping experience. With a commitment to accessibility, 
            our solution not only enhances convenience but also fosters inclusivity, bridging the 
            gap between the visually impaired and the shopping environment. By transforming visual
            information into spoken descriptions, our application revolutionizes the way blind
                individuals engage with their surroundings, promoting autonomy and dignity in 
                everyday activities.
                </div>''', unsafe_allow_html=True)

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image_path = uploaded_image.name
        print(image_path)
        image = Image.open(uploaded_image)

        # st.image(image, caption='Uploaded Image', use_column_width=True)
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

                    # Generate and print random variation
                    phrases = random.choice(variations)
                    print(phrases)
                    convert_to_speech(phrases)
                #     convert_to_speech(predicted_class)

def convert_to_speech(text):
    # Initialize the text-to-speech engine
    tts = gTTS(text,lang='en')
    tts.save('hello.mp3')
    os.system('afplay hello.mp3')            

if __name__ == "__main__":
    main()
