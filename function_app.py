import azure.functions as func
import logging
from intervUAI.main import *
import json
import requests
from PyPDF2 import PdfReader
from io import BytesIO
from intervUAI.main import invoke,resume,summarize
import time
from functools import wraps

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
logging.basicConfig(level=logging.INFO)

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(f"Started '{func.__name__}'")
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logging.info(f"Finished '{func.__name__}' in {elapsed_time:.2f} seconds")
        return result
    return wrapper


@app.route(route="invoke", methods=["POST"])
@log_execution_time
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('HTTP trigger received a request.')
    try:
        use_kb = int(req.params.get("use_kb"))
        try:
            req_body = req.get_json()
        except ValueError:
            logging.info("Request body is not JSON, falling back to form data.")
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON format. Please provide valid JSON."}),
                status_code=400,
                mimetype="application/json"
            )
        
        resume_url = req_body.get("resume","")
        position = req_body.get("position")
        jd_text = req_body.get("jd", "")
        category_percent = req_body.get("category_percent","")
        kb_url = req_body.get("kb_url", "")
        if resume_url != "":
            logging.info(f"Fetching resume from URL: {resume_url}")
            try:
                response = requests.get(resume_url, timeout=10)
                response.raise_for_status()
                pdf = PdfReader(BytesIO(response.content))
                resume_text = "\n".join(
                    [page.extract_text() for page in pdf.pages if page.extract_text()]
                )
                pdf = PdfReader(BytesIO(response.content))
                resume_text = " ".join(
                    [page.extract_text() for page in pdf.pages if page.extract_text()]
                )   
            except Exception as e:
                logging.exception("Failed to fetch or parse resume.")
                raise e
        else:
            resume_text = ""
            
        logging.info("Resume text successfully extracted.")
        logging.info(resume_text)

        ai_message,session_id = invoke(resume_text,position,use_kb,category_percent,jd_text,kb_url)
        return func.HttpResponse(
            json.dumps({"success": True, "message": ai_message, "session": session_id}),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.exception("Error during /invoke request")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="resume")
@log_execution_time
def http_trigger_2(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Resume endpoint hit.')
    try:
        session_id = req.params.get("session_id")
        try:
            req_body = req.get_json()
        except ValueError:  
            logging.info("Request body is not JSON, falling back to form data.")
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON format. Please provide valid JSON."}),
                status_code=400,
                mimetype="application/json"
            )
        
        user_input = req_body.get("user_msg")
    except Exception as e:
        logging.exception("Error extracting resume request parameters.")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )

    try:
        ai_message, session_id = resume(user_input, session_id)
        result = {
            "success": True,
            "message": ai_message,
            "session": session_id
        }
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logging.exception("Error during /resume processing.")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="summarize", methods=["GET"])
@log_execution_time
def http_trigger_summarize(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Summarize endpoint hit.')
    
    try:
        # For GET, extract session_id from query parameters
        if req.method == "GET":
            session_id = req.params.get("session_id")

        if not session_id:
            return func.HttpResponse(
                json.dumps({"error": "session_id is required."}),
                status_code=400,
                mimetype="application/json"
            )

    except Exception as e:
        logging.exception("Error extracting session_id from request.")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )

    try:
        # Call summarize function with session_id
        summary_obj, ratings_obj, session_id = summarize(session_id)
        result = {
            "success": True,
            "summary": summary_obj,
            "ratings": ratings_obj,
            "session": session_id
        }
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.exception("Error during /summarize processing.")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
