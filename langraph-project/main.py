import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from typing import TypedDict, Dict, Any
from fpdf import FPDF
from datetime import datetime
import os
load_dotenv()

class CampaignDetails(BaseModel):
    product: str = Field(description="The agricultural product being sold.")
    quantity: str = Field(description="The quantity of the product available (e.g., '10 kg', '5 quintals').")
    contact_details: str = Field(description="Contact information for the seller.", default="Please contact through the platform.")
    summary: str = Field(description="A brief summary of the post for the advertisement.")

class AgriState(TypedDict):
    user_input: str
    category: str
    response: str
    attempts: int
    draft_answer: str
    verification_result: str
    chat_history: list

llm = ChatOpenAI(temperature=0, model="gpt-4o")
tavily_tool = TavilySearchResults(max_results=5)
tools = [tavily_tool]

# --- Prompt Templates ---
router_prompt = ChatPromptTemplate.from_template(
    """You are an agricultural assistant router. Classify the user's input into one of these categories:
- government_schemes: for questions about financial aid, subsidies, loans, and government programs.
- crop_planner: for crop planning and scheduling related queries.
- campaign: for users wanting to sell/advertise a product.
- crop_health: for questions about plant disease, sick plants, pests, or visual symptoms on crops.
User Input: {user_input}
Provide your response in JSON format with a single key 'category'."""
)

campaign_prompt = ChatPromptTemplate.from_template(
    """You are an expert at extracting information. A user wants to create an ad to sell produce.
Extract the details and format it according to the instructions.
{format_instructions}
User Post: {user_input}"""
)

researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert agricultural schemes researcher for India. Find relevant government schemes for the farmer.
- Use the search tool for the most up-to-date information. Assume the current date is July 2025.
- If no state is specified, assume Karnataka.
- Synthesize findings into a clear response, listing 2-3 relevant schemes with descriptions, benefits, and eligibility.
- If you can't find relevant schemes, state that clearly."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{user_input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

verifier_prompt = ChatPromptTemplate.from_template(
    """You are a verification expert. Given a user's question and a proposed answer, determine if the answer is relevant and accurate for Indian agricultural schemes.
- A 'Good' answer lists specific, real schemes.
- A 'Bad' answer is generic, vague, or hallucinatory.
User Question: {user_input}
Proposed Answer: {draft_answer}
Respond with ONLY 'Good' or 'Bad'."""
)

router_chain = router_prompt | llm | JsonOutputParser()
campaign_extraction_chain = campaign_prompt | llm | PydanticOutputParser(pydantic_object=CampaignDetails)
verifier_chain = verifier_prompt | llm | StrOutputParser()
researcher_agent = create_openai_tools_agent(llm, tools, researcher_prompt)
researcher_agent_executor = AgentExecutor(agent=researcher_agent, tools=tools, verbose=False)

def create_contract_pdf(ad_details: Dict[str, Any]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Agricultural Produce Sale Agreement', border=1, ln=1, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, 'Parties of the Agreement', ln=1)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8,
        f"This agreement is made on {datetime.now().strftime('%B %d, %Y')} between:\n"
        f"The Seller: A Local Farmer from Karnataka, India (contact: {ad_details['contact_details']})\n"
        f"The Buyer: A Registered User of the Agri-Connect Platform."
    )
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, 'Details of the Produce', ln=1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"  - Product: {ad_details['product'].title()}", ln=1)
    pdf.cell(0, 8, f"  - Quantity: {ad_details['quantity']}", ln=1)
    pdf.cell(0, 8, f"  - Price: To be mutually agreed upon by both parties.", ln=1)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, 'Terms & Conditions', ln=1)
    pdf.set_font("Arial", '', 10)
    terms = (
        "1. The quality of the produce shall be consistent with the description provided by the seller.\n"
        "2. Payment shall be made in full upon satisfactory inspection and delivery of the produce.\n"
        "3. This agreement is governed by the local agricultural laws of Karnataka, India.\n"
        "4. Both parties agree to resolve any disputes amicably."
    )
    pdf.multi_cell(0, 6, terms)
    pdf.ln(20)
    pdf.set_font("Arial", '', 12)
    pdf.cell(90, 10, "Seller's Signature:", 0, 0)
    pdf.cell(90, 10, "Buyer's Signature:", 0, 1)
    pdf.cell(90, 10, "____________________", 0, 0)
    pdf.cell(90, 10, "____________________", 0, 1)
    return bytes(pdf.output())

# --- Graph Node Functions ---
def route_node(state: AgriState) -> AgriState:
    result = router_chain.invoke({"user_input": state["user_input"]})
    state["category"] = result["category"]
    state["attempts"] = 0
    state["chat_history"] = []
    return state

def researcher_agent_node(state: AgriState) -> AgriState:
    state["attempts"] += 1
    result = researcher_agent_executor.invoke({
        "user_input": state["user_input"], "chat_history": state["chat_history"]
    })
    state["draft_answer"] = result['output']
    return state

def verifier_node(state: AgriState) -> AgriState:
    result = verifier_chain.invoke({
        "user_input": state["user_input"], "draft_answer": state["draft_answer"]
    })
    state["verification_result"] = result
    return state

def finalize_response_node(state: AgriState) -> AgriState:
    state["response"] = state["draft_answer"]
    return state

def fallback_node(state: AgriState) -> AgriState:
    state["response"] = "I am sorry, but after multiple attempts, I could not find relevant government schemes. Please try rephrasing your question."
    return state

def campaign_node(state: AgriState) -> AgriState:
    extracted_data = campaign_extraction_chain.invoke({
        "user_input": state["user_input"],
        "format_instructions": PydanticOutputParser(pydantic_object=CampaignDetails).get_format_instructions()
    })
    st.session_state.campaigns.append(extracted_data.dict())
    state["response"] = f"✅ Success! Your ad for **{extracted_data.quantity} of {extracted_data.product}** has been posted."
    return state

def crop_planner_node(state: AgriState) -> AgriState:
    state["response"] = "This is a dummy response for crop planning."
    return state
    
def crop_health_node(state: AgriState) -> AgriState:
    state["response"] = "This is a dummy response for crop health."
    return state

def check_verification_and_attempts(state: AgriState) -> str:
    if state["verification_result"] == "Good":
        return "finalize"
    elif state["attempts"] >= 3:
        return "fallback"
    else:
        return "retry"

workflow = StateGraph(AgriState)

workflow.add_node("router", route_node)
workflow.add_node("researcher_agent", researcher_agent_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("fallback", fallback_node)
workflow.add_node("finalize_response", finalize_response_node)
workflow.add_node("campaign", campaign_node)
workflow.add_node("crop_planner", crop_planner_node)
workflow.add_node("crop_health", crop_health_node)

workflow.set_entry_point("router")
workflow.add_edge("researcher_agent", "verifier")
workflow.add_edge("fallback", END)
workflow.add_edge("finalize_response", END)
workflow.add_edge("campaign", END)
workflow.add_edge("crop_planner", END)
workflow.add_edge("crop_health", END)


workflow.add_conditional_edges("router", lambda x: x["category"], {
    "government_schemes": "researcher_agent", "campaign": "campaign",
    "crop_planner": "crop_planner", "crop_health": "crop_health",
})
workflow.add_conditional_edges("verifier", check_verification_and_attempts, {
    "finalize": "finalize_response", "fallback": "fallback", "retry": "researcher_agent",
})

app = workflow.compile()

st.set_page_config(page_title="Agri-Connect", page_icon="🌾", layout="wide")
st.title("🌾 Agri-Connect Assistant & Marketplace")

if 'campaigns' not in st.session_state:
    st.session_state.campaigns = []

seller_tab, buyer_tab = st.tabs(["📢 Ask Assistant / Post Ad", "🛒 Browse Ads"])

with seller_tab:
    st.header("How can I help you today?")
    st.markdown("Ask about government schemes, or post your produce for sale.")

    user_input = st.text_area(
        "Example: 'I am a small farmer in Mandya, what loans are available for buying a tractor?'",
        height=150, key="seller_input"
    )

    if st.button("Process Query", type="primary"):
        if user_input:
            with st.spinner('Processing your query... This may take a moment.'):
                try:
                    initial_state = {
                        "user_input": user_input, "category": "", "response": "", "attempts": 0,
                        "draft_answer": "", "verification_result": "", "chat_history": [],
                    }
                    
                    final_result = None
                    for s in app.stream(initial_state, {"recursion_limit": 5}):
                        final_result = s
                    
                    st.success("Query Processed!")
                    st.markdown(final_result[next(reversed(final_result))]['response'])

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter your question or post.")

with buyer_tab:
    st.header("Available Produce Marketplace")
    st.markdown("Here are the latest ads from our sellers.")
    st.divider()

    if not st.session_state.campaigns:
        st.info("No ads posted yet. Check back later!")
    else:
        for i, ad in enumerate(reversed(st.session_state.campaigns)):
            with st.container(border=True):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader(f"{ad['product'].title()}")
                    st.write(ad['summary'])
                    st.markdown(f"**Contact:** `{ad['contact_details']}`")
                    pdf_bytes = create_contract_pdf(ad)
                    st.download_button(
                        label="📄 Download Agreement", data=pdf_bytes,
                        file_name=f"Contract_{ad['product'].replace(' ', '_')}_{i}.pdf",
                        mime="application/pdf", key=f"download_btn_{i}"
                    )
                with col2:
                    st.metric(label="Quantity Available", value=ad['quantity'])
            st.empty()