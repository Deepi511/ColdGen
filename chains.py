import os
from typing import List, Dict, Union, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from loguru import logger
import json
import re

load_dotenv()

class Job(BaseModel):
    role: str = Field(description="The job title or role")
    experience: str = Field(description="Years of experience or level required")
    skills: List[str] = Field(description="List of required technical skills")
    description: str = Field(description="Brief summary of the job responsibilities")

class Chain:
    def __init__(self):
        # Faster model for structured extraction
        self.llm_fast = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )
        # High-quality model for creative writing
        self.llm_quality = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )
        self.output_parser = JsonOutputParser(pydantic_object=Job)

    def extract_jobs(self, cleaned_text: str) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Extract job postings from scraped page text.
        """
        if not cleaned_text or not cleaned_text.strip():
            logger.error("Empty or invalid text provided for job extraction.")
            raise ValueError("Empty or invalid text provided for job extraction.")

        prompt_extract = PromptTemplate(
            template="""
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format.
            {format_instructions}
            
            If no jobs are found, return an empty array [].
            """,
            input_variables=["page_data"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        
        chain_extract = prompt_extract | self.llm_fast
        
        try:
            res = chain_extract.invoke(input={"page_data": cleaned_text})
            # LangChain's JsonOutputParser often handles the parsing automatically if piped
            # but ChatGroq might return a BaseMessage, so we might need to parse manually if not using a sequential chain
            
            content = res.content if hasattr(res, 'content') else str(res)
            
            # Use the parser to get structured data
            try:
                parsed = self.output_parser.parse(content)
            except Exception:
                # Fallback to manual JSON extraction if specialized parser fails
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                else:
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        parsed = [json.loads(json_match.group(0))]
                    else:
                        parsed = []

            # Normalize and validate output
            if isinstance(parsed, list):
                validated_jobs = []
                for job in parsed:
                    if isinstance(job, dict):
                        validated_job = {
                            'role': str(job.get('role', 'Unknown Role')),
                            'experience': str(job.get('experience', 'Not specified')),
                            'skills': job.get('skills', []) if isinstance(job.get('skills'), list) else [],
                            'description': str(job.get('description', 'No description available'))
                        }
                        validated_jobs.append(validated_job)
                
                logger.info(f"Successfully extracted {len(validated_jobs)} jobs")
                return validated_jobs if validated_jobs else [self._create_fallback_job(cleaned_text)]
            
            elif isinstance(parsed, dict):
                validated_job = {
                    'role': str(parsed.get('role', 'Unknown Role')),
                    'experience': str(parsed.get('experience', 'Not specified')),
                    'skills': parsed.get('skills', []) if isinstance(parsed.get('skills'), list) else [],
                    'description': str(parsed.get('description', 'No description available'))
                }
                logger.info("Successfully extracted 1 job")
                return [validated_job]
            
            return [self._create_fallback_job(cleaned_text)]

        except Exception as e:
            logger.error(f"Error in job extraction: {e}")
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
        Generate a cold email based on a job and matched projects with few-shot examples.
        """
        
        # Handle links safely
        link_str = "No specific projects matched, but I have general experience in AI-based applications."
        if links and isinstance(links, list):
            descriptions = [link['description'] for link in links if isinstance(link, dict) and link.get('description')]
            if descriptions:
                link_str = "\n".join(descriptions)

        job_description = f"Role: {job.get('role')}\nSkills: {', '.join(job.get('skills', []))}\nDescription: {job.get('description')}"

        # Few-shot examples
        examples = """
        Example 1 (Tone: Professional):
        Subject: Strategic AI Implementation for [Company Name]
        Dear Hiring Manager,
        I am [Name], an AI Developer with experience building scalable RAG systems. I noticed your opening for an AI Engineer and was impressed by your work in [Industry]. My portfolio includes a similar project where I optimized document retrieval speed by 40% using ChromaDB. I would love to discuss how my expertise can contribute to your team.
        Best regards, [Name]

        Example 2 (Tone: Friendly):
        Subject: Huge fan of your recent work! 🚀
        Hi team,
        I'm [Name], and I've been following your AI initiatives for a while. Your recent post about [Topic] really resonated with me. I've built a few AI tools myself, including an automated lead generator that matches skills with job requirements perfectly. I'd love to bring that same energy to your [Role] position. Looking forward to chatting!
        Cheers, [Name]
        """

        prompt_email = PromptTemplate.from_template(
            f"""
            ### EXAMPLES FOR REFERENCE:
            {examples}

            ### JOB DESCRIPTION:
            {{job_description}}

            ### INSTRUCTION:
            You are {{username}}, an AI enthusiast. Write a cold email for the job above.
            
            Guidelines:
            - Use a {{tone}} tone
            - reference these relevant projects:
            {{link_list}}
            - Do NOT use placeholders like [Link]. Use the provided project descriptions directly if relevant.
            - Focus on how your portfolio matches the specific skills: {{skills}}
            
            Return only the email content.
            """
        )

        chain_email = prompt_email | self.llm_quality
        
        try:
            logger.info(f"Generating email for {job.get('role')} in {tone} tone")
            res = chain_email.invoke({
                "job_description": job_description,
                "link_list": link_str,
                "username": username,
                "tone": tone,
                "skills": ", ".join(job.get("skills", []))
            })
            
            return res.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating email: {e}")
            return f"Error generating email: {str(e)}"