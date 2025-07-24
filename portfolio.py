import pandas as pd
import chromadb
import uuid
import os
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings

class Portfolio:
    def __init__(self, file_path: str = None):
        # Use relative path or provided path
        if file_path is None:
            file_path = os.path.join("resource", "project_portfolio.csv")
        
        self.file_path = file_path
        self.data = None
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        
        # Initialize components
        self._initialize()

    def _initialize(self):
        """Initialize the portfolio with error handling"""
        try:
            # Load CSV data
            if not os.path.exists(self.file_path):
                print(f"Warning: Portfolio file not found at {self.file_path}")
                self.data = pd.DataFrame(columns=["Techstack", "Description"])
            else:
                self.data = pd.read_csv(self.file_path)
                
                # Validate required columns
                required_columns = ["Techstack", "Description"]
                for col in required_columns:
                    if col not in self.data.columns:
                        print(f"Warning: Missing column '{col}' in portfolio CSV")
                        self.data[col] = "Not specified"
                
                # Clean data
                self.data = self.data.dropna(subset=["Techstack", "Description"])
                self.data["Techstack"] = self.data["Techstack"].astype(str)
                self.data["Description"] = self.data["Description"].astype(str)

            # Initialize embedding model
            try:
                self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            except Exception as e:
                print(f"Error initializing embedding model: {e}")
                self.embedding_model = None
                return

            # Initialize ChromaDB
            try:
                self.chroma_client = chromadb.PersistentClient("vectorstore")
                self.collection = self.chroma_client.get_or_create_collection(
                    name="portfolio"
                )
            except Exception as e:
                print(f"Error initializing ChromaDB: {e}")
                self.chroma_client = None
                self.collection = None

        except Exception as e:
            print(f"Error initializing portfolio: {e}")
            self.data = pd.DataFrame(columns=["Techstack", "Description"])

    def load_portfolio(self) -> bool:
        """Load portfolio data into vector store"""
        try:
            if self.collection is None or self.embedding_model is None:
                print("Portfolio components not properly initialized")
                return False

            if self.data.empty:
                print("No portfolio data available")
                return False

            # Check if collection already has data
            if self.collection.count() > 0:
                print("Portfolio already loaded")
                return True

            # Prepare data for ChromaDB
            documents = self.data["Techstack"].tolist()
            metadatas = [{"description": desc} for desc in self.data["Description"]]
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]

            # Add to collection with error handling
            try:
                # Add documents in batches to avoid memory issues
                batch_size = 100
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i+batch_size]
                    batch_metas = metadatas[i:i+batch_size]
                    batch_ids = ids[i:i+batch_size]
                    
                    self.collection.add(
                        documents=batch_docs,
                        metadatas=batch_metas,
                        ids=batch_ids
                    )
                
                print(f"Successfully loaded {len(documents)} portfolio items")
                return True
                
            except Exception as e:
                print(f"Error adding documents to collection: {e}")
                return False

        except Exception as e:
            print(f"Error loading portfolio: {e}")
            return False

    def query_links(self, skills: List[str]) -> List[Dict[str, Any]]:
        """Query portfolio for relevant projects based on skills"""
        try:
            if not skills or not isinstance(skills, list):
                print("No skills provided for querying")
                return []

            if self.collection is None:
                print("Collection not initialized")
                return []

            # Filter out empty skills
            valid_skills = [skill.strip() for skill in skills if skill and skill.strip()]
            if not valid_skills:
                print("No valid skills provided")
                return []

            # Create query text
            query_text = " ".join(valid_skills)
            
            # Query the collection
            try:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=min(5, max(2, self.collection.count()))  # Adaptive result count
                )
                
                # Extract metadata
                metadatas = results.get('metadatas', [])
                if metadatas and isinstance(metadatas, list):
                    # Flatten if nested
                    if metadatas and isinstance(metadatas[0], list):
                        metadatas = metadatas[0]
                    
                    # Filter out empty or invalid metadata
                    valid_metadatas = []
                    for meta in metadatas:
                        if isinstance(meta, dict) and meta.get('description'):
                            valid_metadatas.append(meta)
                    
                    return valid_metadatas
                
                return []
                
            except Exception as e:
                print(f"Error querying collection: {e}")
                return []

        except Exception as e:
            print(f"Error in query_links: {e}")
            return []

    def get_all_projects(self) -> List[Dict[str, str]]:
        """Get all projects from the portfolio"""
        try:
            if self.data is None or self.data.empty:
                return []
            
            projects = []
            for _, row in self.data.iterrows():
                projects.append({
                    "techstack": str(row["Techstack"]),
                    "description": str(row["Description"])
                })
            
            return projects
            
        except Exception as e:
            print(f"Error getting all projects: {e}")
            return []

    def add_project(self, techstack: str, description: str) -> bool:
        """Add a new project to the portfolio"""
        try:
            if not techstack or not description:
                print("Both techstack and description are required")
                return False

            # Add to dataframe
            new_row = pd.DataFrame({
                "Techstack": [techstack],
                "Description": [description]
            })
            
            self.data = pd.concat([self.data, new_row], ignore_index=True)
            
            # Save to CSV
            try:
                self.data.to_csv(self.file_path, index=False)
            except Exception as e:
                print(f"Error saving to CSV: {e}")
                return False

            # Add to vector store if initialized
            if self.collection is not None:
                try:
                    self.collection.add(
                        documents=[techstack],
                        metadatas=[{"description": description}],
                        ids=[str(uuid.uuid4())]
                    )
                except Exception as e:
                    print(f"Error adding to vector store: {e}")
                    # Continue anyway since CSV is updated

            return True

        except Exception as e:
            print(f"Error adding project: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if portfolio is ready for use"""
        return (self.collection is not None and 
                self.embedding_model is not None and 
                self.data is not None and 
                not self.data.empty)