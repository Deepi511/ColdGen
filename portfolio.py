import pandas as pd
import chromadb
import uuid
import os
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger

class Portfolio:
    def __init__(self, file_path: str = None):
        if file_path is None:
            file_path = os.path.join("resource", "project_portfolio.csv")
        
        self.file_path = file_path
        self.data = None
        self.chroma_client = None
        self.collection = None
        
        self._initialize()

    def _initialize(self):
        """Initialize the portfolio with structured logging"""
        try:
            if not os.path.exists(self.file_path):
                logger.warning(f"Portfolio file not found at {self.file_path}")
                self.data = pd.DataFrame(columns=["Techstack", "Description"])
            else:
                self.data = pd.read_csv(self.file_path)
                
                required_columns = ["Techstack", "Description"]
                for col in required_columns:
                    if col not in self.data.columns:
                        logger.error(f"Missing required column '{col}' in portfolio CSV")
                        self.data[col] = "Not specified"
                
                self.data = self.data.dropna(subset=["Techstack", "Description"])
                self.data["Techstack"] = self.data["Techstack"].astype(str)
                self.data["Description"] = self.data["Description"].astype(str)
                logger.debug(f"Loaded {len(self.data)} items from CSV")

            # ChromaDB handles embeddings internally by default using all-MiniLM-L6-v2
            # No need to load a separate model unless specific customization is needed.

            try:
                self.chroma_client = chromadb.PersistentClient("vectorstore")
                self.collection = self.chroma_client.get_or_create_collection(name="portfolio")
                logger.info("ChromaDB initialized")
            except Exception as e:
                logger.error(f"Error initializing ChromaDB: {e}")
                self.chroma_client = None
                self.collection = None

        except Exception as e:
            logger.exception(f"Critical error during portfolio initialization: {e}")
            self.data = pd.DataFrame(columns=["Techstack", "Description"])

    def load_portfolio(self) -> bool:
        """Load portfolio data into vector store with logging"""
        try:
            if self.collection is None:
                logger.error("Portfolio components not properly initialized")
                return False

            if self.data.empty:
                logger.warning("No portfolio data available to load")
                return False

            if self.collection.count() > 0:
                logger.info(f"Portfolio already contains {self.collection.count()} items")
                return True

            documents = self.data["Techstack"].tolist()
            metadatas = [{"description": desc} for desc in self.data["Description"]]
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]

            batch_size = 100
            for i in range(0, len(documents), batch_size):
                self.collection.add(
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size],
                    ids=ids[i:i+batch_size]
                )
            
            logger.info(f"Successfully loaded {len(documents)} portfolio items into vector store")
            return True

        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            return False

    def query_links(self, skills: List[str]) -> List[Dict[str, Any]]:
        """Query portfolio for relevant projects with adaptive results"""
        try:
            if not skills or not isinstance(skills, list):
                logger.debug("No skills provided for querying")
                return []

            if self.collection is None:
                logger.error("Collection not initialized")
                return []

            valid_skills = [skill.strip() for skill in skills if skill and skill.strip()]
            if not valid_skills:
                return []

            query_text = " ".join(valid_skills)
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(5, max(1, self.collection.count()))
            )
            
            metadatas = results.get('metadatas', [])
            if metadatas and isinstance(metadatas[0], list):
                metadatas = metadatas[0]
            
            valid_metadatas = [meta for meta in metadatas if isinstance(meta, dict) and meta.get('description')]
            logger.info(f"Found {len(valid_metadatas)} relevant portfolio items for skills: {valid_skills[:3]}...")
            return valid_metadatas

        except Exception as e:
            logger.error(f"Error querying portfolio: {e}")
            return []

    def get_all_projects(self) -> List[Dict[str, str]]:
        if self.data is None or self.data.empty:
            return []
        return [{"techstack": str(row["Techstack"]), "description": str(row["Description"])} 
                for _, row in self.data.iterrows()]

    def add_project(self, techstack: str, description: str) -> bool:
        """Add a new project with synchronization logging"""
        try:
            if not techstack or not description:
                logger.error("Both techstack and description are required")
                return False

            new_row = pd.DataFrame({"Techstack": [techstack], "Description": [description]})
            self.data = pd.concat([self.data, new_row], ignore_index=True)
            self.data.to_csv(self.file_path, index=False)
            logger.info(f"Project added to CSV: {techstack[:20]}...")

            if self.collection is not None:
                self.collection.add(
                    documents=[techstack],
                    metadatas=[{"description": description}],
                    ids=[str(uuid.uuid4())]
                )
                logger.info("Project added to vector store")

            return True
        except Exception as e:
            logger.error(f"Error adding project: {e}")
            return False

    def is_ready(self) -> bool:
        return (self.collection is not None and 
                self.data is not None and 
                not self.data.empty)