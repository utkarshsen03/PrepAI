from langgraph.types import interrupt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
import time
from functools import wraps
from datetime import datetime

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"[{datetime.now()}] Starting '{func.__name__}'...")
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"[{datetime.now()}] Finished '{func.__name__}' in {duration:.2f} seconds.")
        
        return result
    return wrapper
@log_execution_time
def human_input_node(state):
    logging.info("human_input_node triggered")
    # state['route'] = "interrupt"
    # print("\nPlease answer the question below:\n")
    ai_msg = state.get("ai_message", "")

    if ai_msg:
        # print(f"IntervU AI: {ai_msg}")
        logging.warning(f"IntervU AI: {ai_msg}")
    else:
        logging.warning("No AI message found in state.")

    user_message = interrupt("Your response: ")
    # logging.info(f"User responded with: {user_message}")

    # if not user_message:
    #     logging.warning("User submitted an empty response.")

    state["user_message"] = user_message
    return state

    # except Exception as e:
    #     logging.exception("Error in human_input_node: %s", str(e))
    #     state["error"] = "An error occurred while receiving user input."
    #     return state
