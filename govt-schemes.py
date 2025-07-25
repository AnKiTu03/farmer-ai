import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)

def government_schemes_analyzer(query: str) -> str:
    """Answers questions about Indian government schemes using the exact prompt from your agent flow."""
    print("\n🤖 Running Government Schemes Analyzer...")
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

if __name__ == "__main__":
    print("--- Government Schemes Analyzer ---")
    query_input = input("Enter your question about a government scheme: ")
    answer_result = government_schemes_analyzer(query_input)
    print("\n--- Response ---")
    print(answer_result)