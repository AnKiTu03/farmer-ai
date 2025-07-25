import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# --- Setup ---
load_dotenv()
# The user's provided code uses a variable `llm_gemini`. We will initialize it here.
llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Pydantic model for the financial plan output, as needed by your provided function
class CropPlanOutput(BaseModel):
    summary: str = Field(description="A brief summary of the financial outlook.")
    estimated_yield: str = Field(description="Estimated yield in tonnes or quintals per acre.")
    estimated_revenue: str = Field(description="Estimated total revenue in INR, with market price assumptions.")
    investment_required: str = Field(description="Breakdown of costs (seeds, fertilizer, labor, etc.) in INR.")
    roi_percent: float = Field(description="The calculated Return on Investment percentage.")

# --- Functions (Adapted from your Agent Flow) ---

def recommend_crops_with_llm(N: int, P: int, K: int, ph: float) -> list[str]:
    """Uses an LLM to recommend top 3 crops based on soil data."""
    print("\n🤖 Running AI Crop Recommender (LLM Step)...")
    prompt = ChatPromptTemplate.from_template(
        """You are an expert agronomist for Bengaluru, Karnataka, India.
        Based on the following soil conditions, recommend the top 3 most suitable crops.
        - Nitrogen (N): {N} kg/ha
        - Phosphorous (P): {P} kg/ha
        - Potassium (K): {K} kg/ha
        - pH: {ph}
        Today's date is July 25, 2025, which is the Kharif season.
        Return ONLY a comma-separated list of the 3 crop names. Example: Maize, Ragi, Cotton"""
    )
    recommender_chain = prompt | llm_gemini
    response = recommender_chain.invoke({"N": N, "P": P, "K": K, "ph": ph})
    # Clean up the response to get a list
    recommended_crops = [crop.strip() for crop in response.content.split(',')]
    return recommended_crops[:3]

def generate_financial_plan(crop_name: str, area_in_acres: float) -> str:
    """Generates a detailed financial plan using the exact prompt from your agent flow."""
    print(f"\n🤖 Generating financial plan for {crop_name}...")
    plan_parser = PydanticOutputParser(pydantic_object=CropPlanOutput)
    plan_prompt = ChatPromptTemplate.from_template(
        """You are an expert agricultural economist for Bengaluru, Karnataka, India.
        The farmer has chosen to cultivate **{crop_name}** on **{area} acres** of land.
        Generate a concise financial plan with realistic estimates for this region. Today is July 25, 2025.
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

# --- Main execution block to get terminal input ---
if __name__ == "__main__":
    print("--- AI Crop & Financial Planner ---")
    n_input = int(input("Enter Nitrogen (N) content (kg/ha): "))
    p_input = int(input("Enter Phosphorous (P) content (kg/ha): "))
    k_input = int(input("Enter Potassium (K) content (kg/ha): "))
    ph_input = float(input("Enter soil pH level (e.g., 6.5): "))
    area_input = float(input("Enter your farm area in acres: "))

    top_3_crops = recommend_crops_with_llm(n_input, p_input, k_input, ph_input)
    print(f"\nBased on your farm's details, here are the top 3 recommended crops:")
    for i, crop in enumerate(top_3_crops):
        print(f"{i+1}. {crop}")

    choice = int(input("\nWhich crop would you like a financial plan for? (Enter 1, 2, or 3): "))
    selected_crop = top_3_crops[choice-1]

    final_plan = generate_financial_plan(selected_crop, area_input)
    print(final_plan)