from langchain_groq import ChatGroq
from intervUAI.configs.prompts import RATING_PROMPT, SUMMARY_PROMPT
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Evaluation:
    def __init__(self):
        try:
            self.llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")
            logging.info("LLM initialized successfully.")
        except Exception as e:
            logging.exception("Failed to initialize LLM: %s", str(e))
            raise

    def rate_answer_node(self, state):
        logging.info("rate_answer_node triggered")

        try:
            # Extract information from state
            jd = state.get("jd", "")
            resume = state.get("resume", "")
            memory = state.get("memory", [])
            ai_message = state.get("ai_message", "")
            user_message = state.get("user_message", "")

            if resume != "":
                resume = "Candidate Resume: " + resume

            logging.info(f"Input state:\n- Question: {ai_message}\n- Answer: {user_message}")

            k = 0
            ratings = []
            while k < len(memory):
                j = k
                k += 20
                chats = memory[j:k]
                prompt = RATING_PROMPT.format(resume=resume, jd=jd, memory=chats)
                logging.debug(f"Constructed RATING_PROMPT:\n{prompt}")
                rating = self.llm.invoke(prompt).content
                ratings.extend(json.loads(rating))
                logging.info(f"Rating generated for memory chunk {j}-{k}:\n{rating}")
            logging.info(f"Ratings generated:\n{ratings}")

            logging.info(f"Received rating:\n{rating}")
            
            # Update state with rating and set end to True
            state["ratings"] = ratings
            state["end"] = True

            return state

        except Exception as e:
            logging.exception("Error in rate_answer_node: %s", str(e))
            state["error"] = "An error occurred during rating."
            return state

    def summarize_node(self, state):
        logging.info("summarize_node triggered")

        try:
            ratings = state.get("ratings", [])
            jd = state.get("jd", "")
            resume = state.get("resume", "")
            memory = state.get("memory", [])

            if resume != "":
                resume = "Candidate Resume: " + resume

            # Log the key input information
            logging.info(f"Summary input state:\n- Ratings count: {len(ratings)}\n- Resume length: {len(resume)}\n- JD length: {len(jd)}")

            # Ensure memory and ratings are in the correct format for the prompt
            if not isinstance(memory, list):
                logging.error("Memory is not a list, it's of type: %s", type(memory))
                state["error"] = "Memory should be a list."
                return state

            # Construct the prompt for summarization
            prompt = SUMMARY_PROMPT.format(memory=str(memory), ratings=str(ratings), resume=resume, jd=jd)
            logging.debug(f"Constructed SUMMARY_PROMPT:\n{prompt}")

            # Call LLM for summarization
            summary = self.llm.invoke(prompt).content
            logging.info(f"Generated summary:\n{summary}")
            
            # Update state with summary
            state['summary'] = summary
            return {"summary": summary, **state}

        except Exception as e:
            logging.exception("Error in summarize_node: %s", str(e))
            state["error"] = "An error occurred during summarization."
            raise Exception(f"Summarization failed: {str(e)}")
