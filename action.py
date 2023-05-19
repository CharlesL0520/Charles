import openai
import pandas as pd

# Set up the OpenAI API key
openai.api_key = '<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>'

# Define a function to generate text using the OpenAI GPT model
def generate_text(prompt):
    model = "text-davinci-002"
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = response.choices[0].text
    return message

# Define a function to read the Excel file
def read_excel_data(file_path):
    data_df = pd.read_excel(file_path)
    data_dict = {}
    for row in data_df.itertuples(index=False):
        if row[0] not in data_dict:
            data_dict[row[0]] = []
        data_dict[row[0]].append(row)
    return data_dict

# Define a function to generate responses based on user input
def generate_excel_response(input_data):
    words = input_data.split()
    column_1_value = words[0]
    data_dict = read_excel_data("path/to/excel/file.xlsx")
    if column_1_value in data_dict:
        row = data_dict[column_1_value]
        prompt = "Column: {}\nRow: {}\nQ: {}\nA:{}"
        response = prompt.format(row[0], row[1], row[2], row[3])
        return response
    else:
        return f"No data found for '{column_1_value}' in the Excel file."

# Define a function to generate responses to user's questions
def generate_response(question, model_id=None):
    prompt = (f"Q: {question}\nA:")
    parameters = {
        "model": "<model-id>" if model_id is None else model_id,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 1024,
        "n": 1,
        "stop": "\n"
    }
    response = openai.Completion.create(**parameters)
    return response.choices[0].text.strip()

# Define a function to train the ChatGPT model
def train_chatbot(data):
    parameters = {
        "model": "text-davinci-002",
        "samples": data,
        "temperature": 0.5,
        "max_epochs": 100,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "validation_split": 0.1
    }
    response = openai.FineTune.create(**parameters)
    return response.id

# Prepare the training data
data = [
    {"prompt": "Q: What services does your logistics company offer?\nA: Our logistics company offers a wide range of services, including transportation, warehousing, inventory management, order fulfillment, and supply chain management. We work with businesses of all sizes to provide customized logistics solutions that meet their unique needs."},
    {"prompt": "Q: How can I track my shipment?\nA: You can track your shipment by logging into your account on our website and entering your shipment tracking number. You can also contact our customer support team for assistance with tracking your shipment."},
    {"prompt": "Q: How long will it take for my shipment to arrive?\nA: The delivery time for your shipment will depend on several factors, including the shipping method you choose, the distance your shipment needs to travel, and any customs or clearance procedures that may be required. We will provide you with an estimated delivery date when you place your order, and we will do our best to ensure that your shipment arrives on time."},
    {"prompt": "Q: How do I place an order for shipping?\nA: You can place an order for shipping by logging into your account on our website and selecting the shipping service that best meets your needs. You can then enter your shipment details and pay for your order online."},
    {"prompt": "Q: Can I ship internationally?\nA: Yes, we offer international shipping services to many countries around the world. However, please note that international shipping may be subject to customs and clearance procedures, which can cause delays in delivery."},
    {"prompt": "Q: What if my shipment is lost or damaged?\nA: If your shipment is lost or damaged, please contact our customer support team immediately. We will work with you to investigate the issue and determine the best course of action to resolve the problem."},
    {"prompt": "Q: How do I cancel or modify my shipment?\nA: To cancel or modify your shipment, please contact our customer support team as soon as possible. We will do our best to accommodate your request, but please note that cancellation or modification may not be possible once your shipment has been picked up by our carrier."},
    #Track
    {"prompt": "Q: Where is my package?\nA: Track Number, please"},
    {"prompt": "Q: Can you tell me the status of my shipment?\nA: Track Number, please"},
    {"prompt": "Q: When will my package arrive?\nA: Track Number please"},
    {"prompt": "Q: What's the latest update on my shipment?\nA: Track Number, please"},
    #Schedule Delivery
    {"prompt": "Q: I want to schedule a delivery for tomorrow.\nA: Please visit the Order history page and select the item you wish to schedule."},
    {"prompt": "Q: When can you deliver my package?\nA: Please visit the Order history page and select the item you wish to schedule."},
    {"prompt": "Q: How can I schedule a delivery?\nA: Please visit the Order history page and select the item you wish to schedule."},
    {"prompt": "Q: What are the available delivery slots for next week?\nA: Please visit the Order history page and select the order you wish to schedule."},
    #Cancel Shipment
    {"prompt": "Q: I want to cancel my shipment.\nA: Tracking Number"},
    {"prompt": "Q: How can I cancel my order?\nA: Tracking Number"},
    {"prompt": "Q: Can you help me cancel my shipment?\nA: Tracking Number"},
    {"prompt": "Q: Is it possible to cancel my package delivery?\nA: Tracking Number"}

]


# Train the ChatGPT model
model_id = train_chatbot(data)
excel_data = read_excel_data("path/to/excel/file.xlsx")
data.extend(excel_data)
message = "your received message here"
data.append({"prompt": "Q: " + message + "\nA:"})
train_chatbot(data)

# Set the ID of the trained model for generating responses
generate_response.__defaults__ = (model_id,)
