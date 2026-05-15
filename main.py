from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio
from utils import clean_text
from loguru import logger
from functools import lru_cache
import os
import traceback

app = Flask(__name__)

# Configure logger
logger.add("app.log", rotation="500 MB")

# Initialize components with error handling
try:
    chain = Chain()
    portfolio = Portfolio()
    # Load portfolio once at startup to avoid overhead during requests
    portfolio.load_portfolio()
    logger.info("Application components initialized successfully")
except Exception as e:
    logger.error(f"Error initializing components: {e}")
    chain = None
    portfolio = None

# Cache for storing last processed data
cached_data = {
    "last_job": None,
    "last_links": None,
    "last_username": None,
    "last_tone": None,
    "last_url": None
}

@lru_cache(maxsize=50)
def get_job_data(url):
    """Cached function to fetch and clean job data from a URL"""
    logger.info(f"Scraping and cleaning data from: {url}")
    loader = WebBaseLoader([url])
    raw_data = loader.load()
    
    if not raw_data:
        return None
    
    return clean_text(raw_data[0].page_content)

@app.route("/", methods=["GET", "POST"])
def index():
    email_result = None
    error = None
    jobs_found = 0

    if request.method == "POST":
        if chain is None or portfolio is None:
            error = "Application components not properly initialized."
            logger.error(error)
            return render_template("index.html", email=email_result, error=error, jobs_found=jobs_found)

        if "generate" in request.form:
            url = request.form.get("job_url", "").strip()
            username = request.form.get("username", "").strip() or "User"
            tone = request.form.get("tone", "formal").strip()

            if not url:
                error = "Please provide a valid job URL."
                return render_template("index.html", email=email_result, error=error, jobs_found=jobs_found)

            try:
                logger.info(f"Processing URL: {url}")
                cleaned_data = get_job_data(url)
                
                if not cleaned_data or len(cleaned_data.strip()) < 50:
                    raise ValueError("No valid content could be extracted from the provided URL.")

                logger.info("Extracting jobs...")
                jobs = chain.extract_jobs(cleaned_data)
                jobs_found = len(jobs) if jobs else 0
                
                if not jobs:
                    raise ValueError("No job postings could be extracted.")

                job = jobs[0]
                skills = job.get("skills", [])
                
                links = []
                if portfolio.is_ready() and skills:
                    links = portfolio.query_links(skills)
                    logger.info(f"Matched {len(links)} portfolio items")

                logger.info("Generating email...")
                email = chain.write_mail(job, links, username=username, tone=tone)
                
                if not email:
                    raise ValueError("Email generation returned empty result.")

                cached_data.update({
                    "last_job": job,
                    "last_links": links,
                    "last_username": username,
                    "last_tone": tone,
                    "last_url": url
                })

                email_result = email
                logger.info("Email generated successfully")

            except Exception as e:
                error = f"Error: {str(e)}"
                logger.error(f"Processing failed: {error}")
                logger.debug(traceback.format_exc())

        elif "regenerate" in request.form:
            if not cached_data["last_job"]:
                error = "No previous data found."
                return render_template("index.html", email=email_result, error=error, jobs_found=jobs_found)

            try:
                logger.info("Regenerating email...")
                email = chain.write_mail(
                    cached_data["last_job"],
                    cached_data["last_links"] or [],
                    username=cached_data["last_username"],
                    tone=cached_data["last_tone"]
                )
                email_result = email
                jobs_found = 1
            except Exception as e:
                error = f"Regeneration failed: {str(e)}"
                logger.error(error)

    return render_template("index.html", email=email_result, error=error, jobs_found=jobs_found)

@app.route("/ping")
def ping():
    return jsonify({"status": "running"}), 200

@app.route("/health")
def health():
    status = "healthy" if chain and portfolio and portfolio.is_ready() else "unhealthy"
    return jsonify({"status": status}), 200 if status == "healthy" else 500

@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template("500.html"), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "True").lower() == "true"
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)