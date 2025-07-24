import os
from typing import List, Dict, Union, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import json
import re

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192"
        )

    def extract_jobs(self, cleaned_text: str) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Extract job postings from scraped page text.

        Returns:
            A list of dictionaries, each representing one job with keys:
            - 'role': str
            - 'experience': str
            - 'skills': List[str]
            - 'description': str
        """
        if not cleaned_text or not cleaned_text.strip():
            raise ValueError("Empty or invalid text provided for job extraction.")

        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills`, and `description`.
            
            Important formatting rules:
            - Return a valid JSON array of job objects
            - Each job must have all four keys: role, experience, skills, description
            - skills should be an array of strings
            - If no jobs found, return an empty array []
            - Do not include any text before or after the JSON
            
            ### VALID JSON (NO PREAMBLE):
            """
        )
        
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})

        try:
            # Clean the response content
            content = res.content.strip()
            
            # Try to extract JSON from the response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Try to find single job object
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = '[' + json_match.group(0) + ']'
                else:
                    json_str = content

            # Parse JSON
            parsed = json.loads(json_str)
            
            # Normalize and validate output
            if isinstance(parsed, list):
                validated_jobs = []
                for job in parsed:
                    if isinstance(job, dict):
                        # Ensure all required keys exist
                        validated_job = {
                            'role': str(job.get('role', 'Unknown Role')),
                            'experience': str(job.get('experience', 'Not specified')),
                            'skills': job.get('skills', []) if isinstance(job.get('skills'), list) else [],
                            'description': str(job.get('description', 'No description available'))
                        }
                        validated_jobs.append(validated_job)
                return validated_jobs if validated_jobs else [self._create_fallback_job(cleaned_text)]
            
            elif isinstance(parsed, dict):
                validated_job = {
                    'role': str(parsed.get('role', 'Unknown Role')),
                    'experience': str(parsed.get('experience', 'Not specified')),
                    'skills': parsed.get('skills', []) if isinstance(parsed.get('skills'), list) else [],
                    'description': str(parsed.get('description', 'No description available'))
                }
                return [validated_job]
            
            else:
                return [self._create_fallback_job(cleaned_text)]

        except (json.JSONDecodeError, OutputParserException, Exception) as e:
            print(f"Error parsing job extraction result: {e}")
            return [self._create_fallback_job(cleaned_text)]

    def _create_fallback_job(self, text: str) -> Dict[str, Union[str, List[str]]]:
        """Create a fallback job when extraction fails"""
        return {
            'role': 'Position Available',
            'experience': 'Not specified',
            'skills': ['General skills required'],
            'description': text[:500] + '...' if len(text) > 500 else text
        }

    def write_mail(
        self,
        job: Dict[str, Union[str, List[str]]],
        links: List[Dict[str, str]],
        username: str = "User",
        tone: str = "formal"
    ) -> str:
        """
        Generate a cold email based on a job and matched projects.

        Args:
            job: A dictionary with job details (role, skills, etc.)
            links: A list of project metadata dictionaries with 'description'
            username: Name to use in email
            tone: 'formal', 'casual', etc.

        Returns:
            The generated email as a string.
        """
        
        # Validate inputs
        if not job or not isinstance(job, dict):
            raise ValueError("Invalid job data provided")
        
        if not username or not username.strip():
            username = "User"
        
        if tone not in ["formal", "casual", "professional", "friendly"]:
            tone = "formal"

        # Handle links safely
        link_str = "No specific projects matched, but I have general experience in AI-based applications."
        if links and isinstance(links, list):
            descriptions = []
            for link in links:
                if isinstance(link, dict) and 'description' in link:
                    desc = link['description']
                    if desc and desc.strip():
                        descriptions.append(desc.strip())
            
            if descriptions:
                link_str = "\n".join(descriptions)

        # Create job description string safely
        job_desc_parts = []
        if job.get('role'):
            job_desc_parts.append(f"Role: {job['role']}")
        if job.get('experience'):
            job_desc_parts.append(f"Experience: {job['experience']}")
        if job.get('skills') and isinstance(job['skills'], list):
            job_desc_parts.append(f"Skills: {', '.join(job['skills'])}")
        if job.get('description'):
            job_desc_parts.append(f"Description: {job['description']}")
        
        job_description = "\n".join(job_desc_parts)

        prompt_email = PromptTemplate.from_template(
            f"""
            ### JOB DESCRIPTION:
            {{job_description}}

            ### INSTRUCTION:
            You are {username}, an AI enthusiast and passionate about building innovative AI applications.
            Write a cold email introducing yourself and how your skills align with the company's needs. 
            
            Guidelines:
            - Use a {tone} tone
            - Keep it concise and professional
            - Focus on relevant experience and skills
            - Do NOT mention programming languages or tools you don't actually know
            - Be genuine and authentic
            
            Use only relevant projects from the following list (if applicable):
            {{link_list}}

            Do not provide a preamble or explanations, just the email content.

            ### EMAIL (NO PREAMBLE):
            """
        )

        chain_email = prompt_email | self.llm
        
        try:
            res = chain_email.invoke({
                "job_description": job_description,
                "link_list": link_str
            })
            
            return res.content.strip() if res.content else "Email generation failed. Please try again."
            
        except Exception as e:
            print(f"Error generating email: {e}")
            return f"Error generating email: {str(e)}"