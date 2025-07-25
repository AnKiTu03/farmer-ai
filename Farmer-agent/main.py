import os
import json
import pickle
from PIL import Image
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from datetime import datetime
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

# Import our Firestore configuration
from firestore_config import get_db

# --- Setup: Load Environment Variables & Initialize DB ---
load_dotenv()
db = get_db() # Initialize Firestore

# --- Pydantic Models for Structured Input/Output ---
class CampaignInput(BaseModel):
    """Input schema for the create_campaign tool."""
    title: str = Field(description="The main title or name of the campaign.")
    crop: str = Field(description="The specific crop being sold, e.g., 'Tomato', 'Wheat'.")
    cropType: str = Field(description="The variety or type of the crop, e.g., 'Heirloom', 'Basmati'.")
    location: str = Field(description="The city or district where the produce is located.")
    duration: str = Field(description="The duration for which the campaign will be active, e.g., '7 days', '2 weeks'.")
    estimatedYield: str = Field(description="The total estimated amount of produce available, e.g., '10 tons', '50 quintals'.")
    minimumQuotation: str = Field(description="The starting price or minimum bid, e.g., '₹1500 per quintal'.")
    notes: str = Field(description="Any additional notes or details about the produce or campaign.", default="")

class FetchFilter(BaseModel):
    """Defines the structure for a single filter condition for fetching documents."""
    field: str = Field(description="The name of the document field to filter on.")
    op: str = Field(description="The comparison operator, e.g., '==', '>', '<', '>=', '<=', 'in'.")
    value: Any = Field(description="The value to compare against.")

class CropPlannerInput(BaseModel):
    """Input schema for the AI Crop Planner tool."""
    N: int = Field(description="Nitrogen (N) value in soil (kg/ha)")
    P: int = Field(description="Phosphorus (P) value in soil (kg/ha)")
    K: int = Field(description="Potassium (K) value in soil (kg/ha)")
    ph: float = Field(description="Soil pH value, from 0.0 to 14.0")
    area_in_acres: float = Field(description="Area of the land in acres")

class FinancialPlanInput(BaseModel):
    """Input schema for the Financial Plan Generation tool."""
    crop_name: str = Field(description="The specific name of the crop selected by the user.")
    area_in_acres: float = Field(description="The area of the land in acres, as provided by the user.")

class CropPlanOutput(BaseModel):
    """Data model for a crop financial plan."""
    estimated_yield: str = Field(description="Projected output for the given land area, in tons or quintals.")
    estimated_revenue: str = Field(description="Gross income expected based on local market prices, in INR.")
    investment_required: str = Field(description="Approximate cost for seeds, fertilizers, labor, etc., in INR.")
    roi_percent: float = Field(description="Return on Investment as a percentage.")
    summary: str = Field(description="A brief summary of the plan and justification for the recommended crop.")

# --- Initialize LLMs ---
llm_openai = ChatOpenAI(temperature=0, model="gpt-4o")
# Using a valid and powerful Gemini model name
llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3, convert_system_message_to_human=True)

# --- Load Local Machine Learning Models ---
working_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Load Crop Recommendation Model (.pkl)
CROP_MODEL_LOADED = False
try:
    # Updated path to your new model file
    crop_model_path = os.path.join(working_dir, "RandomForest-2.pkl")
    with open(crop_model_path, 'rb') as file:
        crop_model = pickle.load(file)
    CROP_MODEL_LOADED = True
    print("✅ Crop recommendation model loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading crop model: {e}")

# 2. Load Disease Classifier Model (TensorFlow)
DISEASE_MODEL_LOADED = False
try:
    model_path = os.path.join(working_dir, "plant_disease_prediction_model.h5")
    class_indices_path = os.path.join(working_dir, "class_indices.json")
    if os.path.exists(model_path) and os.path.exists(class_indices_path):
        disease_model = tf.keras.models.load_model(model_path)
        with open(class_indices_path) as f:
            class_indices = json.load(f)
        DISEASE_MODEL_LOADED = True
        print("✅ Disease classification model loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading disease model: {e}")


# --- Helper Functions for Models ---
def predict_top_3_crops_with_rf(N, P, K, temp, humidity, ph, rainfall):
    """Uses the loaded RandomForest model to predict the top 3 most likely crops."""
    input_data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    probabilities = crop_model.predict_proba(input_data)[0]
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_crops = crop_model.classes_[top_3_indices]
    return top_3_crops

def get_weather_data(location="Bengaluru"):
    """Returns placeholder weather data."""
    return {"temperature": 27.5, "humidity": 75.0, "rainfall": 120.0}

def load_and_preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file).resize(target_size)
    img_array = np.array(img)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_file, class_indices_map):
    preprocessed_img = load_and_preprocess_image(image_file)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_indices_map[str(predicted_class_index)]


# --- Agent Tools Definition ---

@tool(args_schema=CampaignInput)
def create_campaign(title: str, crop: str, cropType: str, location: str, duration: str, estimatedYield: str, minimumQuotation: str, notes: str = "") -> str:
    """
    Creates a new campaign document in the 'campaigns' collection in Firestore.
    Use this tool after you have gathered all the required information from the user.
    """
    if not db:
        return "Error: Firestore is not initialized. Check server logs."
    try:
        print(f"\n🤖 Creating new campaign: {title}...")
        
        # Assemble the data dictionary from the arguments
        data = {
            "title": title,
            "crop": crop,
            "cropType": cropType,
            "location": location,
            "duration": duration,
            "status": "active",  # Default status for new campaigns
            "estimatedYield": estimatedYield,
            "minimumQuotation": minimumQuotation,
            "notes": notes,
            "currentBid": minimumQuotation, # Default current bid to minimum
            "totalBids": 0,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        }

        # Add the document
        _, doc_ref = db.collection("campaigns").add(data)
        print(f"   - Campaign created with ID: {doc_ref.id}")
        return f"Successfully created the campaign '{title}' with ID: {doc_ref.id}"
    except Exception as e:
        print(f"   - Error creating campaign: {e}")
        return f"Error: Could not create the campaign. Details: {e}"

@tool
def fetch_documents_from_firestore(collection_name: str, filters: list[dict] = None, limit: int = None) -> list:
    """
    Fetches documents from a specified Firestore collection (campaigns, bids, orders, contracts).
    Can optionally filter the results and limit the number of documents returned.
    Each filter should be a dictionary with 'field', 'op', and 'value'.
    """
    if not db: return "Error: Firestore is not initialized."
    try:
        print(f"\n🤖 Fetching documents from {collection_name} with filters: {filters} and limit: {limit}...")
        query = db.collection(collection_name)
        if filters:
            for f in filters:
                validated_filter = FetchFilter(**f)
                query = query.where(field_path=validated_filter.field, op_string=validated_filter.op, value=validated_filter.value)
        
        if limit:
            query = query.limit(limit)

        docs = query.stream()
        results = [{"id": doc.id, **doc.to_dict()} for doc in docs]
        return results if results else "No documents found matching the criteria."
    except Exception as e:
        return f"Error: Could not fetch documents. Details: {e}"

@tool(args_schema=CropPlannerInput)
def ai_crop_planner(N: int, P: int, K: int, ph: float, area_in_acres: float) -> str:
    """
    Recommends the top 3 suitable crops based on soil conditions (N, P, K, pH) and local weather.
    This is the first step in creating a crop plan. The output of this tool should be presented to the user
    so they can choose which crop they want a financial plan for.
    """
    if not CROP_MODEL_LOADED:
        return "Error: The crop recommendation model is not available."

    print("\n🤖 Running AI Crop Planner (Recommendation Stage)...")
    weather = get_weather_data()
    recommended_crops = predict_top_3_crops_with_rf(N, P, K, weather['temperature'], weather['humidity'], ph, weather['rainfall'])

    return f"Based on your farm's details, here are the top 3 recommended crops: 1. {recommended_crops[0]}, 2. {recommended_crops[1]}, and 3. {recommended_crops[2]}. Which one would you like me to create a detailed financial plan for?"

@tool(args_schema=FinancialPlanInput)
def generate_financial_plan(crop_name: str, area_in_acres: float) -> str:
    """
    Generates a detailed financial plan for a specific crop and land area.
    This tool should only be used after the user has selected a crop from the options provided by the `ai_crop_planner` tool.
    """
    print(f"\n🤖 Generating financial plan for {crop_name}...")
    plan_parser = PydanticOutputParser(pydantic_object=CropPlanOutput)
    plan_prompt = ChatPromptTemplate.from_template(
        """You are an expert agricultural economist for Bengaluru, Karnataka, India.
        The farmer has chosen to cultivate **{crop_name}** on **{area} acres** of land.
        Generate a concise financial plan with realistic estimates for this region. Today is July 2025.
        {format_instructions}"""
    )
    plan_chain = plan_prompt | llm_gemini | plan_parser
    financial_plan = plan_chain.invoke({
        "crop_name": crop_name,
        "area": area_in_acres,
        "format_instructions": plan_parser.get_format_instructions()
    })

    return f"""
    **Financial Plan for {crop_name}**
    Here is the detailed forecast for cultivating {crop_name} on {area_in_acres} acres:
    - **Summary:** {financial_plan.summary}
    - **Estimated Yield:** {financial_plan.estimated_yield}
    - **Estimated Revenue:** {financial_plan.estimated_revenue}
    - **Investment Required:** {financial_plan.investment_required}
    - **Return on Investment (ROI):** {financial_plan.roi_percent:.2f}%
    """

@tool
def crop_health_analyzer(image_path: str) -> str:
    """
    Analyzes an image of a plant to identify diseases. Takes a local file path to the image as input.
    """
    if not DISEASE_MODEL_LOADED:
        return "Error: The disease classification model is not available."
    if not os.path.exists(image_path):
        return f"Error: The file path '{image_path}' does not exist."

    print("\n🤖 Running Crop Health Analyzer...")
    prediction = predict_image_class(disease_model, image_path, class_indices)
    disease_name = prediction.replace('___', ' ').replace('_', ' ')
    print(f"   - Model predicted disease: {disease_name}")

    explanainer_prompt = ChatPromptTemplate.from_template(
        """A plant has been identified with the disease: **{disease}**.
        Provide a helpful response including: a brief description, symptoms, treatment steps, and preventive measures."""
    )
    explanainer_chain = explanainer_prompt | llm_gemini | StrOutputParser()
    explanation = explanainer_chain.invoke({"disease": disease_name})
    return f"**Model Prediction:** {disease_name}\n\n**Detailed Analysis & Recommendations:**\n{explanation}"

@tool
def government_schemes_analyzer(query: str) -> str:
    """
    Answers questions about Indian government schemes for farmers by referencing its internal knowledge
    of the official myscheme.gov.in and schemes.vikaspedia.in websites.
    """
    print("\n🤖 Running Government Schemes Analyzer (Direct LLM Call)...")
    analysis_prompt = ChatPromptTemplate.from_template(
        """You are an expert on Indian government schemes for farmers, based ONLY on your knowledge from
        https://www.myscheme.gov.in/ and https://schemes.vikaspedia.in/.
        Answer the following farmer's query. If you don't know, say that you couldn't find a relevant scheme on the specified portals.
        Farmer's Query: "{query}"
        Detailed Answer:"""
    )
    analysis_chain = analysis_prompt | llm_gemini | StrOutputParser()
    answer = analysis_chain.invoke({"query": query})
    return answer

# --- Main Agent Setup ---
tools = [
    create_campaign, # Replaced the generic save tool
    fetch_documents_from_firestore,
    ai_crop_planner,
    generate_financial_plan,
    crop_health_analyzer,
    government_schemes_analyzer
]

agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful and conversational agricultural assistant.
        Your goal is to assist farmers by using your available tools.

        **Campaign Management:**
        - To create a new campaign, you MUST use the `create_campaign` tool.
        - Before using the tool, you MUST gather all the required information from the user. The required arguments for the tool are:
          - `title`
          - `crop`
          - `cropType`
          - `location`
          - `duration`
          - `estimatedYield`
          - `minimumQuotation`
          - `notes` (this one is optional)
        - To find or view a campaign, use the `fetch_documents_from_firestore` tool. You can filter by any field, such as `title` or `status`.

        **Viewing Orders, Bids, and Contracts:**
        - To view **orders**, use the `fetch_documents_from_firestore` tool with `collection_name` 'orders'. Fetch the top 5 by setting the `limit` parameter to 5. DO NOT ask for a campaign ID for orders.
        - To view **bids** or **contracts**, you MUST FIRST ask the user "For which campaign ID?" and then use that ID to filter the results with the `fetch_documents_from_firestore` tool.
        

        **Crop Planning (Two-Step Process):**
        1. First, use `ai_crop_planner` after getting N, P, K, pH, and land area from the user. This tool gives 3 options.
        2. Present these 3 options. After the user chooses one, use `generate_financial_plan` to provide the detailed forecast.
        
        Always be friendly and guide the user through the process."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


main_agent = create_openai_tools_agent(llm_openai, tools, agent_prompt)
agent_executor = AgentExecutor(agent=main_agent, tools=tools, verbose=True)

# --- Conversational Loop ---
def run_conversation():
    chat_history = []
    print("🤖 Hello! I am your Agri-Connect assistant. How can I help you today?")
    print("   You can ask me to 'create a campaign', 'get a crop plan', or ask about schemes and diseases.")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("🤖 Goodbye! Have a great day.")
                break
            result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
            print("\nAssistant:", result["output"])
            chat_history.extend([HumanMessage(content=user_input), AIMessage(content=result["output"])])
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_conversation()