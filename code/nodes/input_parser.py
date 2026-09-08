from langchain_community.document_loaders import PyMuPDFLoader

def parse_inputs(resume_path, jd_path):
    loader = PyMuPDFLoader(resume_path)
    resume = "\n".join([doc.page_content for doc in loader.load()])
    with open(jd_path, 'r') as f:
        jd = f.read()
    return resume, jd
