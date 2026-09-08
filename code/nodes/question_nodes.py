from langchain_groq import ChatGroq
from intervUAI.configs.prompts import INTERVIEWER_PROMPT
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

def ask_question_node(state):
    logging.info("ask_question_node triggered")

    try:
        llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature = 0.4)
        logging.info("LLM initialized successfully.")

        resume = state.get("resume", "")
        jd = state.get("jd", "")
        memory = state.get("memory", [])
        user_message = state.get("user_message")
        position = state.get("position")

        if resume != "":
            resume = "Candidate Resume: " + resume

        # category_percent = state.get("category_percent")
        # if category_percent is not None or category_percent != "":
        #     category_percent = "These are the percentage of questions in each category: \n" + str(category_percent)

        logging.info(f"Current state includes resume length: {len(resume)}, jd length: {len(jd)}, position: {position}")

        if not memory:
            # First question
            memory = [
                {"role": "assistant", "content": "Tell me about yourself."}
            ]
            state["memory"] = memory
            state["ai_message"] = "Tell me about yourself."
            logging.info("First question generated.")
            return state

        if user_message:
            memory.append({"role": "user", "content": user_message})
            logging.info(f"Appended user message to memory: {user_message}")
        else:
            logging.warning("No user_message found. Returning early.")
            state["ai_message"] = ""
            state["memory"] = memory
            return state
        
        system_prompt = INTERVIEWER_PROMPT.format(resume=resume, jd=jd, position=position,memory=memory)
        response = llm.invoke(system_prompt)
        ai_reply = response.content
        logging.info(f"AI reply generated: {ai_reply}")

        memory.append({"role": "assistant", "content": ai_reply})
        state["memory"] = memory
        state["ai_message"] = ai_reply
        state["cross_count"] = 0

        return state

    except Exception as e:
        logging.exception("Error in ask_question_node: %s", str(e))
        state["error"] = "An error occurred while generating the question."
        raise Exception("Error in ask_question_node") from e
