"""
MediMemory MCP Server - Medical Image Analysis Extension
Extends SuperMemory MCP Server for medical imaging workflows with AI-powered decision support.
"""

import asyncio
import json
import logging
import uuid
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import base64
from mcp.server import Server
from mcp.types import Tool, TextContent
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MedicalImageMemory:
    """Represents a medical image memory entry"""
    id: str
    image_id: str
    patient_id: str
    study_date: datetime
    modality: str  # e.g., "chest_xray", "ct", "mri"
    image_embedding: List[float]
    text_embedding: Optional[List[float]]
    report_text: str
    pathology_labels: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime
    importance_score: float = 1.0

class MedicalImageEmbedder:
    """CNN-based medical image feature extractor"""

    def __init__(self, model_name="densenet121", device="cpu"):
        self.device = device
        self.model_name = model_name

        # Load pre-trained DenseNet-121 (simulated for CheXpert)
        if model_name == "densenet121":
            self.model = models.densenet121(pretrained=True)
            # Replace classifier with identity for feature extraction
            self.model.classifier = nn.Identity()
            self.feature_dim = 1024

        self.model.eval()
        self.model.to(self.device)

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def embed_image(self, image_path_or_bytes) -> np.ndarray:
        """Extract feature embedding from medical image"""
        try:
            if isinstance(image_path_or_bytes, str):
                image = Image.open(image_path_or_bytes)
            else:
                image = Image.open(io.BytesIO(image_path_or_bytes))

            # Preprocess image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.cpu().numpy().flatten()

            # L2 normalize
            features = features / np.linalg.norm(features)

            return features

        except Exception as e:
            logger.error(f"Error embedding image: {e}")
            # Return random embedding as fallback (for demo)
            return np.random.normal(0, 0.1, self.feature_dim)

class PathologyClassifier:
    """Simulated pathology classification model"""

    def __init__(self):
        self.pathologies = [
            "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", 
            "Infiltration", "Mass", "Nodule", "Pneumonia", 
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", 
            "Fibrosis", "Pleural_Thickening", "Hernia"
        ]

        # Simulated classification weights (in practice, load trained model)
        self.weights = {p: np.random.normal(0, 0.1, 1024) for p in self.pathologies}

    def predict_pathologies(self, image_embedding: np.ndarray) -> Dict[str, float]:
        """Predict pathology probabilities from image embedding"""
        predictions = {}
        for pathology, weight in self.weights.items():
            # Simulated prediction using dot product + sigmoid
            score = np.dot(image_embedding, weight)
            prob = 1 / (1 + np.exp(-score))  # Sigmoid
            predictions[pathology] = float(prob)

        return predictions

class MediMemoryStore:
    """Medical image memory store with vector and metadata storage"""

    def __init__(self, db_path: str = "medimemory.db"):
        self.db_path = db_path
        self.image_embedder = MedicalImageEmbedder()
        self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.pathology_classifier = PathologyClassifier()

        # Initialize FAISS index for multi-modal embeddings
        self.embedding_dim = 1024 + 384  # Image + text embeddings
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Initialize database
        self.init_database()
        self.load_memories()

    def init_database(self):
        """Initialize SQLite database for medical image metadata"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS medical_memories (
                id TEXT PRIMARY KEY,
                image_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                study_date TEXT NOT NULL,
                modality TEXT NOT NULL,
                image_embedding TEXT,
                text_embedding TEXT,
                report_text TEXT,
                pathology_labels TEXT,
                metadata TEXT,
                timestamp TEXT,
                importance_score REAL,
                embedding_id INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def load_memories(self):
        """Load existing medical memories from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('SELECT * FROM medical_memories ORDER BY timestamp')

        embeddings = []
        for row in cursor:
            # Reconstruct multi-modal embedding
            image_emb = json.loads(row[5]) if row[5] else [0] * 1024
            text_emb = json.loads(row[6]) if row[6] else [0] * 384
            combined_emb = image_emb + text_emb
            embeddings.append(combined_emb)

        if embeddings:
            embeddings_array = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)

        conn.close()
        logger.info(f"Loaded {len(embeddings)} medical memories from database")

    def store_medical_memory(self, 
                           image_data: bytes,
                           patient_id: str,
                           study_date: datetime,
                           modality: str,
                           report_text: str = "",
                           metadata: Dict[str, Any] = None) -> str:
        """Store medical image memory with AI analysis"""

        # Generate embeddings
        image_embedding = self.image_embedder.embed_image(image_data)
        text_embedding = self.text_embedder.encode([report_text])[0] if report_text else np.zeros(384)

        # Predict pathologies
        pathology_predictions = self.pathology_classifier.predict_pathologies(image_embedding)
        predicted_pathologies = [p for p, score in pathology_predictions.items() if score > 0.5]

        # Create memory entry
        memory_id = str(uuid.uuid4())
        image_id = f"img_{memory_id[:8]}"

        memory = MedicalImageMemory(
            id=memory_id,
            image_id=image_id,
            patient_id=patient_id,
            study_date=study_date,
            modality=modality,
            image_embedding=image_embedding.tolist(),
            text_embedding=text_embedding.tolist(),
            report_text=report_text,
            pathology_labels=predicted_pathologies,
            metadata=metadata or {},
            timestamp=datetime.now(timezone.utc)
        )

        # Combine embeddings for storage
        combined_embedding = np.concatenate([image_embedding, text_embedding])
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

        # Add to FAISS index
        embedding_id = self.index.ntotal
        self.index.add(np.array([combined_embedding]).astype('float32'))

        # Store in database
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO medical_memories 
            (id, image_id, patient_id, study_date, modality, image_embedding, 
             text_embedding, report_text, pathology_labels, metadata, timestamp, 
             importance_score, embedding_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.id,
            memory.image_id,
            memory.patient_id,
            memory.study_date.isoformat(),
            memory.modality,
            json.dumps(memory.image_embedding),
            json.dumps(memory.text_embedding),
            memory.report_text,
            json.dumps(memory.pathology_labels),
            json.dumps(memory.metadata),
            memory.timestamp.isoformat(),
            memory.importance_score,
            embedding_id
        ))
        conn.commit()
        conn.close()

        logger.info(f"Stored medical memory: {memory_id} for patient {patient_id}")
        return memory_id

    def search_similar_cases(self, 
                           query_image: Optional[bytes] = None,
                           query_text: Optional[str] = None,
                           patient_id: Optional[str] = None,
                           modality: Optional[str] = None,
                           pathology: Optional[str] = None,
                           limit: int = 10,
                           threshold: float = 0.7) -> List[MedicalImageMemory]:
        """Search for similar medical cases"""

        if query_image or query_text:
            # Embedding-based search
            if query_image:
                img_emb = self.image_embedder.embed_image(query_image)
            else:
                img_emb = np.zeros(1024)

            if query_text:
                txt_emb = self.text_embedder.encode([query_text])[0]
            else:
                txt_emb = np.zeros(384)

            query_embedding = np.concatenate([img_emb, txt_emb])
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

            # Search FAISS index
            scores, indices = self.index.search(
                np.array([query_embedding]).astype('float32'), 
                min(limit, self.index.ntotal)
            )

            # Filter by threshold and retrieve memories
            memories = []
            conn = sqlite3.connect(self.db_path)

            for score, idx in zip(scores[0], indices[0]):
                if score < threshold:
                    continue

                cursor = conn.execute(
                    'SELECT * FROM medical_memories WHERE embedding_id = ?', (int(idx),)
                )
                row = cursor.fetchone()
                if row:
                    memory = self._row_to_memory(row)
                    memories.append(memory)

            conn.close()

        else:
            # Metadata-based search
            conn = sqlite3.connect(self.db_path)
            query_parts = []
            params = []

            if patient_id:
                query_parts.append("patient_id = ?")
                params.append(patient_id)
            if modality:
                query_parts.append("modality = ?")
                params.append(modality)
            if pathology:
                query_parts.append("pathology_labels LIKE ?")
                params.append(f'%"{pathology}"%')

            if query_parts:
                query = f"SELECT * FROM medical_memories WHERE {' AND '.join(query_parts)} LIMIT ?"
                params.append(limit)
            else:
                query = "SELECT * FROM medical_memories ORDER BY timestamp DESC LIMIT ?"
                params = [limit]

            cursor = conn.execute(query, params)
            memories = [self._row_to_memory(row) for row in cursor.fetchall()]
            conn.close()

        return memories

    def _row_to_memory(self, row) -> MedicalImageMemory:
        """Convert database row to MedicalImageMemory object"""
        return MedicalImageMemory(
            id=row[0],
            image_id=row[1],
            patient_id=row[2],
            study_date=datetime.fromisoformat(row[3]),
            modality=row[4],
            image_embedding=json.loads(row[5]) if row[5] else [],
            text_embedding=json.loads(row[6]) if row[6] else [],
            report_text=row[7],
            pathology_labels=json.loads(row[8]) if row[8] else [],
            metadata=json.loads(row[9]) if row[9] else {},
            timestamp=datetime.fromisoformat(row[10]),
            importance_score=row[11]
        )

class MediMemoryMCPServer:
    """MCP Server for medical image analysis"""

    def __init__(self):
        self.memory_store = MediMemoryStore()
        self.server = Server("medimemory")
        self.setup_handlers()

    def setup_handlers(self):
        """Setup MCP handlers for medical image operations"""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            return [
                Tool(
                    name="store_medical_image",
                    description="Store medical image with AI analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_data": {"type": "string", "description": "Base64 encoded image"},
                            "patient_id": {"type": "string", "description": "Patient identifier"},
                            "study_date": {"type": "string", "description": "Study date (ISO format)"},
                            "modality": {"type": "string", "description": "Imaging modality"},
                            "report_text": {"type": "string", "description": "Radiology report"},
                            "metadata": {"type": "object", "description": "Additional metadata"}
                        },
                        "required": ["image_data", "patient_id", "study_date", "modality"]
                    }
                ),
                Tool(
                    name="search_similar_cases",
                    description="Search for similar medical cases",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query_image": {"type": "string", "description": "Base64 encoded query image"},
                            "query_text": {"type": "string", "description": "Text query"},
                            "patient_id": {"type": "string", "description": "Filter by patient"},
                            "modality": {"type": "string", "description": "Filter by modality"},
                            "pathology": {"type": "string", "description": "Filter by pathology"},
                            "limit": {"type": "integer", "default": 10},
                            "threshold": {"type": "number", "default": 0.7}
                        }
                    }
                ),
                Tool(
                    name="get_diagnosis_suggestions",
                    description="Get AI diagnosis suggestions for image",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_data": {"type": "string", "description": "Base64 encoded image"},
                            "patient_history": {"type": "string", "description": "Patient medical history"}
                        },
                        "required": ["image_data"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name == "store_medical_image":
                image_data = base64.b64decode(arguments["image_data"])
                patient_id = arguments["patient_id"]
                study_date = datetime.fromisoformat(arguments["study_date"])
                modality = arguments["modality"]
                report_text = arguments.get("report_text", "")
                metadata = arguments.get("metadata", {})

                memory_id = self.memory_store.store_medical_memory(
                    image_data, patient_id, study_date, modality, report_text, metadata
                )

                return [TextContent(
                    type="text",
                    text=f"Medical image stored successfully with ID: {memory_id}"
                )]

            elif name == "search_similar_cases":
                query_image = base64.b64decode(arguments["query_image"]) if arguments.get("query_image") else None
                query_text = arguments.get("query_text")
                patient_id = arguments.get("patient_id")
                modality = arguments.get("modality")
                pathology = arguments.get("pathology")
                limit = arguments.get("limit", 10)
                threshold = arguments.get("threshold", 0.7)

                similar_cases = self.memory_store.search_similar_cases(
                    query_image, query_text, patient_id, modality, pathology, limit, threshold
                )

                result = f"Found {len(similar_cases)} similar cases:\n\n"
                for case in similar_cases:
                    result += f"Patient: {case.patient_id}\n"
                    result += f"Date: {case.study_date.strftime('%Y-%m-%d')}\n"
                    result += f"Modality: {case.modality}\n"
                    result += f"Pathologies: {', '.join(case.pathology_labels)}\n"
                    result += f"Report: {case.report_text[:100]}...\n\n"

                return [TextContent(type="text", text=result)]

            elif name == "get_diagnosis_suggestions":
                image_data = base64.b64decode(arguments["image_data"])
                patient_history = arguments.get("patient_history", "")

                # Extract features and predict pathologies
                image_embedding = self.memory_store.image_embedder.embed_image(image_data)
                predictions = self.memory_store.pathology_classifier.predict_pathologies(image_embedding)

                # Find similar cases
                similar_cases = self.memory_store.search_similar_cases(
                    query_image=image_data, limit=5, threshold=0.6
                )

                result = "AI Diagnosis Suggestions:\n\n"
                result += "Pathology Predictions:\n"
                for pathology, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]:
                    result += f"- {pathology}: {prob:.3f}\n"

                result += f"\nSimilar Cases Found: {len(similar_cases)}\n"
                for case in similar_cases[:3]:
                    result += f"- Patient {case.patient_id}: {', '.join(case.pathology_labels)}\n"

                return [TextContent(type="text", text=result)]

            else:
                raise ValueError(f"Unknown tool: {name}")

# FastAPI application
app = FastAPI(title="MediMemory MCP Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global server instance
mcp_server_instance = MediMemoryMCPServer()

@app.post("/upload_medical_image")
async def upload_medical_image(
    file: UploadFile = File(...),
    patient_id: str = "",
    modality: str = "",
    report_text: str = ""
):
    """HTTP endpoint to upload medical image"""
    try:
        image_data = await file.read()
        study_date = datetime.now()

        memory_id = mcp_server_instance.memory_store.store_medical_memory(
            image_data, patient_id, study_date, modality, report_text
        )

        return {"success": True, "memory_id": memory_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/search_cases")
async def search_cases(
    patient_id: str = None,
    modality: str = None,
    pathology: str = None,
    limit: int = 10
):
    """HTTP endpoint to search medical cases"""
    try:
        cases = mcp_server_instance.memory_store.search_similar_cases(
            patient_id=patient_id,
            modality=modality,
            pathology=pathology,
            limit=limit
        )

        return {
            "success": True,
            "cases": [
                {
                    "id": case.id,
                    "patient_id": case.patient_id,
                    "study_date": case.study_date.isoformat(),
                    "modality": case.modality,
                    "pathologies": case.pathology_labels,
                    "report": case.report_text
                }
                for case in cases
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "medimemory-mcp"}

async def main():
    """Main entry point"""
    async with mcp_server_instance.server.request_context():
        logger.info("MediMemory MCP Server started")

        config = uvicorn.Config(app, host="0.0.0.0", port=8001, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
