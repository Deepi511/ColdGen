from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio
from utils import clean_text
import os
import traceback

app = Flask(__name__)

# Initialize components with error handling
try:
    chain = Chain()
    portfolio = Portfolio()
    print("Components initialized successfully")
except Exception as e:
    print(f"Error initializing components: {e}")
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

@app.route("/", methods=["GET", "POST"])
def index():
    email_result = None
    error = None
    jobs_found = 0

    if request.method == "POST":
        # Check if components are initialized
        if chain is None or portfolio is None:
            error = "Application components not properly initialized. Please check your configuration."
            return render_template("index.html", email=email_result, error=error, jobs_found=jobs_found)

        if "generate" in request.form:
            url = request.form.get("job_url", "").strip()
            username = request.form.get("username", "").strip()
            tone = request.form.get("tone", "formal").strip()

            # Validate inputs
            if not url:
                error = "Please provide a valid job URL."
                return render_template("index.html", email=email_result, error=error, jobs_found=jobs_found)

            if not username:
                username = "User"

            if tone not in ["formal", "casual", "professional", "friendly"]:
                tone = "formal"

            try:
                # Load and process webpage
                print(f"Loading URL: {url}")
                loader = WebBaseLoader([url])
                raw_data = loader.load()
                
                if not raw_data:
                    raise ValueError("No content could be loaded from the provided URL.")
                
                # Clean the text
                cleaned_data = clean_text(raw_data[0].page_content)
                
                if not cleaned_data or len(cleaned_data.strip()) < 50:
                    raise ValueError("Insufficient content extracted from the webpage. Please check if the URL contains job postings.")

                # Load portfolio
                print("Loading portfolio...")
                portfolio_loaded = portfolio.load_portfolio()
                if not portfolio_loaded:
                    print("Warning: Portfolio could not be loaded, continuing with empty portfolio")

                # Extract jobs
                print("Extracting jobs...")
                jobs = chain.extract_jobs(cleaned_data)
                jobs_found = len(jobs) if jobs else 0
                
                if not jobs:
                    raise ValueError("No job postings could be extracted from the webpage. Please verify the URL contains job listings.")

                # Use the first job
                job = jobs[0]
                print(f"Found job: {job.get('role', 'Unknown')}")

                # Get skills and query portfolio
                skills = job.get("skills", [])
                if not isinstance(skills, list):
                    skills = []

                print(f"Skills found: {skills}")
                
                links = []
                if portfolio.is_ready() and skills:
                    links = portfolio.query_links(skills)
                    print(f"Found {len(links)} relevant projects")

                # Generate email
                print("Generating email...")
                email = chain.write_mail(job, links, username=username, tone=tone)
                
                if not email or email.strip() == "":
                    raise ValueError("Email generation failed. Please try again.")

                # Cache the data for regeneration
                cached_data.update({
                    "last_job": job,
                    "last_links": links,
                    "last_username": username,
                    "last_tone": tone,
                    "last_url": url
                })

                email_result = email
                print("Email generated successfully")

            except Exception as e:
                error = f"Error processing job posting: {str(e)}"
                print(f"Error: {error}")
                print(traceback.format_exc())

        elif "regenerate" in request.form:
            # Regenerate email with cached data
            if not all([cached_data["last_job"], cached_data["last_username"], cached_data["last_tone"]]):
                error = "No previous data found. Please generate an email first."
                return render_template("index.html", email=email_result, error=error, jobs_found=jobs_found)

            try:
                print("Regenerating email with cached data...")
                email = chain.write_mail(
                    cached_data["last_job"],
                    cached_data["last_links"] or [],
                    username=cached_data["last_username"],
                    tone=cached_data["last_tone"]
                )
                
                if not email or email.strip() == "":
                    raise ValueError("Email regeneration failed. Please try again.")
                
                email_result = email
                jobs_found = 1  # We know we have at least one job in cache
                print("Email regenerated successfully")

            except Exception as e:
                error = f"Regeneration failed: {str(e)}"
                print(f"Regeneration error: {error}")

    return render_template("index.html", email=email_result, error=error, jobs_found=jobs_found)

@app.route("/ping")
def ping():
    """Health check endpoint"""
    status = {
        "status": "running",
        "components": {
            "chain": chain is not None,
            "portfolio": portfolio is not None and portfolio.is_ready()
        }
    }
    return jsonify(status), 200

@app.route("/health")
def health():
    """Detailed health check"""
    health_status = {
        "status": "healthy",
        "components": {
            "chain_initialized": chain is not None,
            "portfolio_initialized": portfolio is not None,
            "portfolio_ready": portfolio.is_ready() if portfolio else False,
            "groq_api_key": os.getenv("GROQ_API_KEY") is not None
        }
    }
    
    # Check if critical components are missing
    if not chain or not portfolio:
        health_status["status"] = "unhealthy"
        return jsonify(health_status), 500
    
    return jsonify(health_status), 200

@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template("500.html"), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "True").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)