"""
MediMemory Client SDK
Python client for interacting with MediMemory MCP Server
"""

import requests
import asyncio
import aiohttp
import base64
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from PIL import Image
import io

@dataclass
class MedicalImageMemory:
    """Medical image memory representation for client"""
    id: str
    image_id: str
    patient_id: str
    study_date: datetime
    modality: str
    pathology_labels: List[str]
    report_text: str
    metadata: Dict[str, Any]
    importance_score: float = 1.0

class MediMemoryClient:
    """Synchronous client for MediMemory MCP Server"""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def upload_medical_image(self,
                           image_path: str,
                           patient_id: str,
                           modality: str,
                           report_text: str = "",
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Upload medical image for analysis and storage"""
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # Prepare form data
            files = {'file': ('image.png', image_data, 'image/png')}
            data = {
                'patient_id': patient_id,
                'modality': modality,
                'report_text': report_text
            }

            response = self.session.post(
                f"{self.base_url}/upload_medical_image",
                files=files,
                data=data
            )
            response.raise_for_status()

            result = response.json()
            if result["success"]:
                return result["memory_id"]
            else:
                raise Exception(f"Upload failed: {result}")

        except Exception as e:
            raise Exception(f"Error uploading image: {e}")

    def search_similar_cases(self,
                           patient_id: Optional[str] = None,
                           modality: Optional[str] = None,
                           pathology: Optional[str] = None,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar medical cases"""
        params = {'limit': limit}
        if patient_id:
            params['patient_id'] = patient_id
        if modality:
            params['modality'] = modality
        if pathology:
            params['pathology'] = pathology

        response = self.session.get(f"{self.base_url}/search_cases", params=params)
        response.raise_for_status()

        result = response.json()
        if result["success"]:
            return result["cases"]
        else:
            raise Exception(f"Search failed: {result}")

    def get_diagnosis_suggestions(self, image_path: str, patient_history: str = "") -> Dict[str, Any]:
        """Get AI diagnosis suggestions for medical image"""
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode()

            # Use MCP interface (simplified for demo)
            payload = {
                "tool": "get_diagnosis_suggestions",
                "arguments": {
                    "image_data": image_base64,
                    "patient_history": patient_history
                }
            }

            # This would normally go through MCP protocol
            # For demo, return simulated suggestions
            suggestions = {
                "pathology_predictions": {
                    "Pneumonia": 0.85,
                    "Effusion": 0.32,
                    "Cardiomegaly": 0.18,
                    "No Finding": 0.12
                },
                "similar_cases": [
                    {"patient_id": "PAT_00123", "pathologies": ["Pneumonia", "Consolidation"]},
                    {"patient_id": "PAT_00456", "pathologies": ["Pneumonia"]}
                ],
                "confidence": 0.85
            }

            return suggestions

        except Exception as e:
            raise Exception(f"Error getting suggestions: {e}")

    def get_patient_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get complete imaging history for a patient"""
        return self.search_similar_cases(patient_id=patient_id, limit=50)

class AsyncMediMemoryClient:
    """Asynchronous client for MediMemory MCP Server"""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip('/')
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def search_similar_cases(self,
                                 patient_id: Optional[str] = None,
                                 modality: Optional[str] = None,
                                 pathology: Optional[str] = None,
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """Async search for similar medical cases"""
        params = {'limit': limit}
        if patient_id:
            params['patient_id'] = patient_id
        if modality:
            params['modality'] = modality
        if pathology:
            params['pathology'] = pathology

        async with self.session.get(f"{self.base_url}/search_cases", params=params) as response:
            response.raise_for_status()
            result = await response.json()

            if result["success"]:
                return result["cases"]
            else:
                raise Exception(f"Search failed: {result}")

# Example usage and demo application
def demo_radiology_workflow():
    """Demonstrate typical radiology workflow with MediMemory"""
    print("🏥 MediMemory Radiology Workflow Demo")
    print("=" * 50)

    client = MediMemoryClient()

    try:
        print("📊 1. Searching for pneumonia cases...")
        pneumonia_cases = client.search_similar_cases(pathology="Pneumonia", limit=5)
        print(f"   Found {len(pneumonia_cases)} pneumonia cases")

        print("\n🔍 2. Searching patient history...")
        patient_cases = client.search_similar_cases(patient_id="PAT_00123", limit=10)
        print(f"   Found {len(patient_cases)} cases for patient PAT_00123")

        print("\n🤖 3. Getting AI diagnosis suggestions...")
        # This would use actual image file in practice
        suggestions = client.get_diagnosis_suggestions("sample_xray.png")
        print("   Top predictions:")
        for pathology, prob in suggestions["pathology_predictions"].items():
            if prob > 0.3:
                print(f"     - {pathology}: {prob:.2%}")

        print(f"\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("   (This is expected when server is not running)")

if __name__ == "__main__":
    demo_radiology_workflow()
