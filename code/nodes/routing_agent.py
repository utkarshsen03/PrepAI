from langchain_groq import ChatGroq
from intervUAI.configs.prompts import ROUTER_PROMPT
import logging
import json

logging.basicConfig(level=logging.INFO)

def route_node(state):
    try:
        logging.info("route_node triggered")

        llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")
        # format = str({    
        # "response": " whether to cross question, next question or end interview", 
        # "thinking process": " How you reached to this conclusion."
        # })
        question = state.get("ai_message", "")
        answer = state.get("user_message", "")
        jd = state.get("jd", "")
        resume = state.get("resume", "")
        history = state.get("memory",[])

        if resume != "":
            resume = "Candidate Resume: " + resume

        prompt = ROUTER_PROMPT.format(question=question, answer=answer, jd=jd, resume=resume, history=history)
        
        try:
            result = llm.invoke(prompt).content.lower()
            logging.info(f"routing decision llm result: {result}")
        except Exception as e:
            logging.error(f"LLM invocation failed: {e}")
            return {**state, "route": "ask"}  # fallback decision
        
        end_flag = True
        try:
            result = json.loads(result)
            logging.info("json load succeded")
            decision = result['response']
            if "cross" in decision and state.get("cross_count", 0) < 2:
                logging.info("decided to cross question")
                return {**state, "route": "cross", "cross_count": state.get("cross_count", 0) + 1}

            if "end" in decision and end_flag:
                logging.info("decided to rate and end interview")
                return {**state, "route": "rate", "end": True}
        except:
            logging.info("json load failed")

            end_flag = False
            logging.warning("LLM Response not in JSON format")
        
        if "cross" in result and state.get("cross_count", 0) < 2:
            logging.info("decided to cross question")
            return {**state, "route": "cross", "cross_count": state.get("cross_count", 0) + 1}

        if "end" in result and end_flag:
            logging.info("decided to rate and end interview")
            return {**state, "route": "rate", "end": True}

        logging.info("moving on to asking question")
        return {**state, "route": "ask"}

    except Exception as e:
        logging.error(f"route_node failed: {e}")
        return {**state, "route": "ask"}  # fail-safe fallback
