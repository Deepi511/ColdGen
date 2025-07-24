import re
from typing import Optional

def clean_text(text: str) -> str:
    """
    Clean and normalize text from web scraping.
    
    Args:
        text: Raw text from webpage
        
    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove script and style content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Keep alphanumeric, spaces, and common tech symbols
    text = re.sub(r'[^a-zA-Z0-9\s.#+\-(),:;]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove extra spaces and trim
    text = text.strip()
    
    return text

def extract_skills_from_text(text: str) -> list:
    """
    Extract potential skills/technologies from text.
    
    Args:
        text: Text to extract skills from
        
    Returns:
        List of potential skills
    """
    if not text:
        return []
    
    # Common tech skills patterns
    tech_patterns = [
        r'\b(?:Python|Java|JavaScript|React|Node\.js|Angular|Vue\.js|Django|Flask|Spring|Express)\b',
        r'\b(?:MySQL|PostgreSQL|MongoDB|Redis|SQLite|Oracle|SQL Server)\b',
        r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Git|GitHub|GitLab|Jenkins|CI/CD)\b',
        r'\b(?:HTML|CSS|SCSS|SASS|Bootstrap|Tailwind|Material-UI|jQuery)\b',
        r'\b(?:Machine Learning|AI|Deep Learning|TensorFlow|PyTorch|Scikit-learn|Pandas|NumPy)\b',
        r'\b(?:REST|API|GraphQL|Microservices|Agile|Scrum|DevOps|Testing|TDD|BDD)\b'
    ]
    
    skills = []
    for pattern in tech_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        skills.extend(matches)
    
    # Remove duplicates and return
    return list(set(skills))

def validate_url(url: str) -> bool:
    """
    Validate if a URL is properly formatted.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None

def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length allowed
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def format_job_description(job: dict) -> str:
    """
    Format job dictionary into readable string.
    
    Args:
        job: Job dictionary
        
    Returns:
        Formatted job description
    """
    if not job or not isinstance(job, dict):
        return "No job information available"
    
    parts = []
    
    if job.get('role'):
        parts.append(f"Role: {job['role']}")
    
    if job.get('experience'):
        parts.append(f"Experience: {job['experience']}")
    
    if job.get('skills') and isinstance(job['skills'], list):
        skills_str = ', '.join(job['skills'])
        parts.append(f"Skills: {skills_str}")
    
    if job.get('description'):
        desc = truncate_text(job['description'], 500)
        parts.append(f"Description: {desc}")
    
    return '\n'.join(parts)

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    if not filename:
        return "untitled"
    
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Ensure it's not empty
    if not filename:
        filename = "untitled"
    
    return filename

def extract_company_name(url: str) -> Optional[str]:
    """
    Extract company name from URL.
    
    Args:
        url: Company URL
        
    Returns:
        Company name or None
    """
    if not url or not isinstance(url, str):
        return None
    
    try:
        # Extract domain
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if domain_match:
            domain = domain_match.group(1)
            # Remove common suffixes
            domain = re.sub(r'\.(com|org|net|edu|gov|io|co\.uk|co\.in)$', '', domain)
            # Capitalize first letter
            return domain.capitalize()
        return None
    except:
        return None