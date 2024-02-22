import pandas as pd
from openai import OpenAI
import json
import time

def categorize_question(question_input, openKey):
    content = f"""Please categorize the following questions into ONE of the provided question types: 
    'object identification', 'activity recognition', 'sport identification',
    'color', 'shape', 'counting', 'location and spatial relations', 'time and sequence', 'person identification',
    'text and signage recognition', 'emotion and sentiment', 'comparison', 'weather', 'animal'.

    Question: What color are the clouds?
    Question Type: color
    
    Question: What is the person doing?
    Question Type: activity recognition
    
    Question: How many people are there?
    Question Type: counting
    
    Question: {question_input}
    Question Type:
    """
    client = OpenAI( api_key=openKey)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )

    response = response.choices[0].message.content
    # Extract the Question Type from the response
    question_type = response.split("Question Type: ")[-1].strip()
    
    return question_type

def main():
    with open('/home/ndhuynh/openai.txt', 'r') as file:
        for line in file:
            openKey = line
    
    question_paths = [
        "/home/ndhuynh/data/simpsonsvqa/v1_Question_Train_simpsons_vqa.json"
        ]

    question_list = []
    for ques_path in question_paths:
        with open(ques_path, 'r') as file:
            questions = json.load(file)["questions"]
            question_list += questions
    question_df = pd.DataFrame(question_list)

    question_unique_list = question_df["question"].unique()

    for i in range(25291, len(question_unique_list)):
        

        while True:
            try:
                question_str = question_unique_list[i]
                chatgpt_question_type = categorize_question(question_str, openKey)

                # Create a dictionary with the current key-value pair
                data = {"question": question_str,
                        "question_type":chatgpt_question_type}

                # Specify the file name where you want to save the JSON data
                file_name = "train_question_type_gpt.json"
                # Open the file in append mode and save the current item to it in JSON format
                with open(file_name, 'a') as json_file:
                    json.dump(data, json_file)
                    json_file.write('\n')  # Add a newline to separate each item

                print(f'[{i + 1}/{len(question_unique_list)}]  to {file_name}')
                break  # Exit the retry loop if successful
            except Exception as e:
                print(f'Error: {e}')
                print('Retrying...')
                time.sleep(2)

if __name__ == "__main__":
    main()