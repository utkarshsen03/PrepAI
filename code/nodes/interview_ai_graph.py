from langgraph.graph import StateGraph
from intervUAI.nodes.question_nodes import ask_question_node
from intervUAI.nodes.cross_question import cross_question_node
from intervUAI.nodes.human_input import human_input_node
from intervUAI.nodes.routing_agent import route_node
from intervUAI.nodes.check_kb_route import check_knowledge_node
from intervUAI.nodes.kb_question_node import ask_kb_question_node
from intervUAI.nodes.final_evaluation import Evaluation
from typing import List, Dict, Optional, TypedDict
import os
import logging
from dotenv import load_dotenv

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class StateInput(TypedDict):
    user_message: str
    memory: List[Dict[str, str]]
    cross_count: int
    resume: str
    jd: Optional[str]
    question: Optional[str]
    answer: Optional[str]
    route: Optional[str]
    end: Optional[bool]
    ratings: List[Dict[str, str]]
    ai_message: str
    tot_ques: int
    summary: str
    position: Optional[str]
    knowledge_base: Optional[str]
    check_kb: Optional[str]
    category_percent: Optional[Dict[str, float]]

def build_interview_graph(checkpointer):
    logging.info("Starting to build the interview graph...")
    
    try:
        builder = StateGraph(StateInput)
        eval = Evaluation()

        # ─────── Nodes ───────
        builder.add_node("check_knowledge", check_knowledge_node)
        builder.add_node("ask_kb_question", ask_kb_question_node)
        builder.add_node("ask_question", ask_question_node)
        builder.add_node("cross_question", cross_question_node)
        builder.add_node("human_input", human_input_node)
        builder.add_node("router_node", route_node)
        builder.add_node("rate_answer", eval.rate_answer_node)
        builder.add_node("summarize", eval.summarize_node)
        logging.info("All nodes added to the graph.")

        # ─────── Edges ───────
        builder.add_edge("ask_kb_question", "human_input")
        builder.add_edge("ask_question", "human_input")
        builder.add_edge("cross_question", "human_input")
        builder.add_edge("human_input", "router_node")
        
        def check_knowledge_route(state):
            return "ask_kb_question" if state.get("check_kb") == 1 else "ask_question"

        builder.add_conditional_edges("check_knowledge", check_knowledge_route)


        def router_transition(state):
            logging.debug("Routing decision started.")
            if state.get("check_kb") == 1 and (len(state.get("memory", [])) -1) // 2 < state.get('tot_ques') +1:
                logging.info("Routing to ask_kb_question.")
                return "ask_kb_question"
            elif state.get("route") == "cross" and state.get("cross_count", 0) < 2:
                logging.info("Routing to cross_question.")
                return "cross_question"
            elif state.get("end") or state.get("route") == "rate" or (len(state.get("memory", []))-1) // 2 >= state.get('tot_ques') + 1:
                logging.info("Routing to rate_answer.")
                state["end"] = True
                return "__end__"
            logging.info("Routing to ask_question.")
            return "ask_question"

        builder.add_conditional_edges("router_node", router_transition)
        builder.add_edge("rate_answer", "summarize")
        builder.add_edge("summarize", "__end__")
        builder.set_entry_point("check_knowledge")
        logging.info("Edges and entry point configured.")

        # ─────── Checkpointer Setup ───────
        conn_str = os.getenv("SQLSERVER_CONN_STRING")
        if not conn_str:
            logging.error("Missing SQLSERVER_CONN_STRING in environment variables.")
            raise RuntimeError("Missing SQLSERVER_CONN_STRING in .env")

        # checkpointer = AzureSQLSaver(conn_str)
        logging.info("AzureSQLSaver initialized.")

        compiled_graph = builder.compile(checkpointer=checkpointer)
        logging.info("Interview graph compiled successfully.")
        return compiled_graph

    except Exception as e:
        logging.exception("Error building the interview graph: %s", str(e))
        raise e
