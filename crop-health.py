import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from PIL import Image

load_dotenv()
llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
vision_model = ChatGoogleGenerativeAI(model="gemini-pro-vision")


def crop_health_analyzer(image_path: str) -> str:
    """Analyzes a crop image using a vision model and then explains the result with your prompt."""
    if not os.path.exists(image_path):
        return f"Error: The file path '{image_path}' does not exist."

    print("\n🤖 Running Crop Health Analyzer...")
    print("   - Identifying disease from image...")
    try:
        img = Image.open(image_path)
    except Exception as e:
        return f"Error opening image file: {e}"

    vision_prompt = "Analyze this image of a plant leaf. Identify the primary disease. Respond with ONLY the common name of the disease. For example: 'Tomato Late Blight' or 'Healthy'."
    disease_name_response = vision_model.invoke(input=[{"type": "text", "text": vision_prompt}, {"type": "image_url", "image_url": img}])
    disease_name = disease_name_response.content.strip()
    print(f"   - Vision model predicted disease: {disease_name}")

    if "healthy" in disease_name.lower():
        return f"**Analysis Result:** The plant appears to be **Healthy**."

    print("   - Generating detailed analysis and recommendations...")
    explanainer_prompt = ChatPromptTemplate.from_template(
        """A plant has been identified with the disease: **{disease}**.
        Provide a helpful response including: a brief description, symptoms, treatment steps, and preventive measures."""
    )
    explanainer_chain = explanainer_prompt | llm_gemini | StrOutputParser()
    explanation = explanainer_chain.invoke({"disease": disease_name})
    
    return f"**Model Prediction:** {disease_name}\n\n**Detailed Analysis & Recommendations:**\n{explanation}"

if __name__ == "__main__":
    print("--- Crop Health Analyzer ---")
    path_input = input("Enter the full path to the crop image file: ")
    analysis_result = crop_health_analyzer(path_input)
    print(analysis_result)