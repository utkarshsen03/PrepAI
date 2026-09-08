# graph.py

import os
from langchain_core.runnables.graph import MermaidDrawMethod
from nodes.interview_ai_graph import build_interview_graph

# Dummy resume and JD to build the graph (mocked)
dummy_resume = "Experienced software developer with AI/ML expertise."
dummy_jd = "Looking for an AI engineer with experience in NLP, LangChain, and LLMs."

# Build and compile the graph
app = build_interview_graph(resume_text=dummy_resume, jd_text=dummy_jd)

def save_graph_image(filename: str = "graph.png"):
    graph_bytes = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    with open(filename, "wb") as f:
        f.write(graph_bytes)
    print(f"Graph saved as {filename}")

if __name__ == "__main__":
    save_graph_image("graph.png")
