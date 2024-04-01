import torch
import transformers
import textwrap
import pandas as pd
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
from sklearn.metrics import accuracy_score
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import re


def create_dataset(path):
    df = pd.read_csv(path, sep = ',')
    dataset_data = [
        {
            "input": row_dict["sentence"],
            "output": row_dict["label_name"],
            "prediction": ''
        }


        for row_dict in df.to_dict(orient="records")
    ]
 
    dataset = pd.DataFrame(dataset_data)
    return dataset

def update_template_few_shot(input_string):
    template = [
        {"role": "user", "content": "Task: In this task, you will be provided with a list of frames and a sentence. Your goal is to select the single most suitable frame from the given list for the provided sentence. Frames are cognitive structures that help humans interpret information by providing a mental framework for understanding. Each frame represents a specific perspective, context, or interpretation. Frame Selection Format: In your response, please select and mention only the name of the most suitable frame from the provided list. Frames List and Definitions: {'Quality of Life': 'Threats and opportunities for the individual\'s wealth, happiness, and well-being.', 'Health and Safety': 'Access to healthcare, health outcomes, disease, sanitation, mental health, violence prevention, infrastructure safety, and public health.', 'Economic': 'The financial consequences and economic implications of the matter on various levels (person, family, community or broader economy).', 'External Regulation and Reputation': 'The external relations of nations or groups, trade agreements, policy outcomes, and external perceptions or consequences.', 'Political': 'Political considerations, actions, efforts, stances, and partisan, bipartisan, or lobbying activities related to an issue.', 'Public Sentiment': 'References of attitudes and opinions of the general public, including polling and demographics.', 'Cultural Identity': 'Traditions, customs, or values of a social group in relation to a policy issue.', 'Security and Defense': 'Actions or calls to action aimed at protecting individuals, groups, or nations from potential threats to their well-being.', 'Policy Prescription and Evaluation': 'Specific policies proposed to address identified issues and the assessment of policy effectiveness.', 'Crime and Punishment': 'Effectiveness and implications of laws and their enforcement.', 'Legality, Constitutionality, Jurisdiction': 'Discusses rights, freedoms, and authority of individuals, corporations, and government.', 'Fairness and Equality': 'The balance or distribution of laws, rights, and resources among individuals or groups.', 'Morality': 'Perspectives, policy objectives, or actions driven by religious principles, duties, ethics, or social responsibilities.', 'Capacity and Resources': 'The presence or absence of various resources (physical, geographic, human, and financial) and the ability of existing systems.', 'Other': 'Frames that don\'t fit into the categories above.'}"},
        {"role": "assistant", "content": "Understood. Can you show me some examples?"},
        {"role": "user", "content": "Input: 'The city lost an estimated 100,000 jobs after the attacks'. What is the frame?"},
        {"role": "assistant", "content": "Economic"},
        {"role": "user", "content": "Input: 'Christmas as many have come to know it in the United States --- dragging home a tree or pulling a fake one out of a box, the frenzy of last-minute shopping, stuffing cards into envelopes --- is different in many immigrant communities in metro Atlanta'. What is the frame?"},
        {"role": "assistant", "content": "Cultural Identity"},
        {"role": "user", "content": "Please select the most appropriate frame for the given sentence, considering the nuances and various perspectives it entails. Make your selection based on how well the frame aligns with the meaning and context of the sentence. Your choice should contribute to a clear and accurate understanding of the sentence's intended interpretation. You are allowed to select only one frame, and your response should be concise and straightforward. Remember, the accuracy of your frame selection is crucial for effective communication and understanding."},
        {"role": "assistant", "content": "Understood. Give me the Input sentence."},
        {"role": "user", "content": "Input: '" + input_string + "'. What is the frame?"}
    ]
    return template


def update_template_zero_shot(input_string):
    template = [
        {"role": "user", "content": "Task: In this task, you will be provided with a list of frames and a sentence. Your goal is to select the single most suitable frame from the given list for the provided sentence. Frames are cognitive structures that help humans interpret information by providing a mental framework for understanding. Each frame represents a specific perspective, context, or interpretation. Frame Selection Format: In your response, please select and mention only the name of the most suitable frame from the provided list. Frames List and Definitions: {'Quality of Life': 'Threats and opportunities for the individual\'s wealth, happiness, and well-being.', 'Health and Safety': 'Access to healthcare, health outcomes, disease, sanitation, mental health, violence prevention, infrastructure safety, and public health.', 'Economic': 'The financial consequences and economic implications of the matter on various levels (person, family, community or broader economy).', 'External Regulation and Reputation': 'The external relations of nations or groups, trade agreements, policy outcomes, and external perceptions or consequences.', 'Political': 'Political considerations, actions, efforts, stances, and partisan, bipartisan, or lobbying activities related to an issue.', 'Public Sentiment': 'References of attitudes and opinions of the general public, including polling and demographics.', 'Cultural Identity': 'Traditions, customs, or values of a social group in relation to a policy issue.', 'Security and Defense': 'Actions or calls to action aimed at protecting individuals, groups, or nations from potential threats to their well-being.', 'Policy Prescription and Evaluation': 'Specific policies proposed to address identified issues and the assessment of policy effectiveness.', 'Crime and Punishment': 'Effectiveness and implications of laws and their enforcement.', 'Legality, Constitutionality, Jurisdiction': 'Discusses rights, freedoms, and authority of individuals, corporations, and government.', 'Fairness and Equality': 'The balance or distribution of laws, rights, and resources among individuals or groups.', 'Morality': 'Perspectives, policy objectives, or actions driven by religious principles, duties, ethics, or social responsibilities.', 'Capacity and Resources': 'The presence or absence of various resources (physical, geographic, human, and financial) and the ability of existing systems.', 'Other': 'Frames that don\'t fit into the categories above.'}. Please select the most appropriate frame for the given sentence, considering the nuances and various perspectives it entails. Make your selection based on how well the frame aligns with the meaning and context of the sentence. Your choice should contribute to a clear and accurate understanding of the sentence's intended interpretation. You are allowed to select only one frame, and your response should be concise and straightforward. Remember, the accuracy of your frame selection is crucial for effective communication and understanding."},
        {"role": "assistant", "content": "Understood. Give me the Input sentence."},
        {"role": "user", "content": "Input: '" + input_string + "'. What is the frame?"}
    ]
    return template

def extract_frame(data):
    phrase = "What is the frame? [/INST]"

    index = data.rfind(phrase)

    if index == -1:
        return None

    start = index + len(phrase)
    end = data.find('.', start)
    frame = data[start:end].strip()

    return frame


if "__main__":
    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser(description="Access a cell in a DataFrame.")
    parser.add_argument('--input_file_path', type=str, help='path of the input file')
    parser.add_argument('--out_file_name', type=str, help='path of the input file')
    parser.add_argument('--learning_type', type=str, help='zero_shot/few_shot')    
    parser.add_argument('--model', type=str, help='mistral/olmo...')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    path = args.input_file_path
    dataset = create_dataset(path)
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.1" # or args.model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    
    for i in range(len(dataset)):
        ipt = dataset.loc[i, 'input']
        if 'zero' in args.learning_type:
            messages = update_template_zero_shot(ipt)
        elif 'few' in args.learning_type:
            messages = update_template_few_shot(ipt)
            
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(device)
        
        generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)
        response = extract_frame(decoded[0])
        print(decoded[0])
        dataset.loc[i, 'prediction'] = response
        
    filename = args.out_file_name
    dataset.to_csv(filename, index=False)
   