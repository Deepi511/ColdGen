# Updated app.py with username, tone, and feedback
from flask import Flask, render_template, request, redirect, url_for
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio
from utils import clean_text

app = Flask(__name__)
chain = Chain()
portfolio = Portfolio()

# To store results between requests
cached_data = {
    "last_job": None,
    "last_links": None,
    "last_username": None,
    "last_tone": None
}

@app.route("/", methods=["GET", "POST"])
def index():
    email_result = None
    error = None

    if request.method == "POST" and "generate" in request.form:
        url = request.form.get("job_url")
        username = request.form.get("username")
        tone = request.form.get("tone")

        try:
            loader = WebBaseLoader([url])
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            jobs = chain.extract_jobs(data)

            job = jobs[0]  # Assume single job listing
            skills = job.get("skills", [])
            links = portfolio.query_links(skills)
            email = chain.write_mail(job, links, username=username, tone=tone)

            # Save for regeneration if needed
            cached_data.update({
                "last_job": job,
                "last_links": links,
                "last_username": username,
                "last_tone": tone
            })

            email_result = email

        except Exception as e:
            error = str(e)

    elif request.method == "POST" and "regenerate" in request.form:
        try:
            email = chain.write_mail(
                cached_data["last_job"],
                cached_data["last_links"],
                username=cached_data["last_username"],
                tone=cached_data["last_tone"]
            )
            email_result = email
        except Exception as e:
            error = str(e)

    return render_template("index.html", email=email_result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
