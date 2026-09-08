from intervUAI.nodes.interview_ai_graph import build_interview_graph
from intervUAI.nodes.load_kb import  fetch_knowledge_data
from intervUAI.nodes.input_parser import parse_inputs
from langchain_core.runnables import RunnableConfig
from intervUAI.checkpointer.redis_checkpointer import RedisSaver
from intervUAI.nodes.final_evaluation import Evaluation
from langgraph.types import Command
from uuid import uuid4
import sys
import json
import logging
import os

def invoke(resume_text,position,use_kb,category_percent,jd_text,kb_url):
    with RedisSaver.from_conn_info(host=os.getenv("host"), port=6380, db=0) as checkpointer:
        try:
            graph = build_interview_graph(checkpointer)

            user_session_id = uuid4()
            
            config = {"configurable": {"thread_id": str(user_session_id)}}

            if use_kb == 1:
                que, kb_base = fetch_knowledge_data(kb_url)
                initial_state = {
                    "user_message":"",
                    "memory":[],
                    "cross_count":0,
                    "resume":resume_text,
                    "position":position,
                    "ratings":[],                    
                    "summary":"",
                    "tot_ques": que,
                    "knowledge_base": kb_base,
                    "category_percent": category_percent
                }
            else:
                initial_state = {
                    "user_message":"",
                    "memory":[],
                    "cross_count":0,
                    "resume":resume_text,
                    "jd":jd_text,
                    "ratings":[],
                    "position":position,
                    "tot_ques": 100,
                    "summary":"",
                    "category_percent": category_percent
                }

            final_state = graph.invoke(initial_state,config=config)

            return final_state['ai_message'],str(user_session_id)
        except Exception as e:
            return str(e),"sdas"

def resume(user_input, session_id):
    with RedisSaver.from_conn_info(host=os.getenv("host"), port=6380, db=0) as checkpointer:
        # try:
            graph = build_interview_graph(checkpointer)
        
            human_command = Command(resume=user_input)
            config = {"configurable": {"thread_id": str(session_id)}}
            final_state= graph.invoke(input=human_command,config=config)

            logging.info(f"Final state after invoking graph: {final_state['ai_message']}")

            if final_state.get("end") or final_state.get("route") == "rate" or (len(final_state.get("memory", []))-1) // 2 >= final_state.get('tot_ques') + 1:

                result = {}

                result["end"] = final_state.get("end",True)
                return (json.dumps(result)), str(session_id)
        
            return(final_state['ai_message']),str(session_id)
        # except Exception as e:
        #     tb = sys.exc_info()[2]
        #     # raise e
        #     return str(e),e.with_traceback(tb)
 
def summarize(session_id):
    try:
        # Connect to Redis
        with RedisSaver.from_conn_info(host=os.getenv("host"), port=6380, db=0) as checkpointer:
            # Load the last saved checkpoint state
            config = {"configurable": {"thread_id": str(session_id)}}

            state = checkpointer.get_tuple(RunnableConfig(config))

            if not state:
                return "No state found for the given session ID.", "",str(session_id)

            # Initialize the Evaluation class
            evaluator = Evaluation()

            # Run summarization
            state = state.checkpoint['channel_values']
            logging.info(state)

            summary_result = evaluator.summarize_node(state)
            rating_result = evaluator.rate_answer_node(state)

            # Extract and return summary
            summary = summary_result.get("summary", "No summary generated.")
            ratings = rating_result.get("ratings", "No rating generated.")

            return {"summary": json.loads(summary)}, {"ratings": ratings}, str(session_id)

    except Exception as e:
        return json.dumps({"error": f"Summarization failed: {str(e)}"}), "", str(session_id)