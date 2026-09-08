from langchain_groq import ChatGroq
from intervUAI.configs.prompts import KB_INTERVIEWER_PROMPT
from dotenv import load_dotenv
import os
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

def ask_kb_question_node(state):
    logging.info("ask_kb_question_node triggered")

    try:
        llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")
        logging.info("LLM initialized successfully.")

        # resume = state.get("resume", "")
        memory = state.get("memory", [])
        user_message = state.get("user_message")
        kb_data = state.get("knowledge_base")
        position = state.get("position")
        # percent_category = state.get("category_percent")
        tot_ques = state.get("tot_ques") + 1
        # if percent_category is not None or percent_category != "":
        #     percent_category = "These are the percentage of questions in each category: \n" + str(percent_category)


        # logging.info(f"Current state includes resume length: {len(resume)}")

        if not memory:
            # First question
            system_prompt = KB_INTERVIEWER_PROMPT.format(knowledge_base=kb_data,position=position,
                                                        total_questions=tot_ques)
            memory = [
                {"role": "system", "content": system_prompt},
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

        response = llm.invoke(memory)
        ai_reply = response.content
        logging.info(f"AI reply generated: {ai_reply}")

        memory.append({"role": "assistant", "content": ai_reply})
        state["memory"] = memory
        state["ai_message"] = ai_reply
        state["cross_question"] = 0

        return state

    except Exception as e:
        logging.exception("Error in ask_question_node: %s", str(e))
        state["error"] = "An error occurred while generating the question."
        return state
