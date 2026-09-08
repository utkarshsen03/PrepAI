def check_knowledge_node(state):
    if state.get("knowledge_base"):
        state["check_kb"] = 1
    else:
        state["check_kb"] = 0
    return state
