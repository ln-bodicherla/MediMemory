# MediMemory: Medical Image Analysis Extension of SuperMemory MCP Server

🏥 **A research/demo Model Context Protocol (MCP) server that stores medical images as embeddings and retrieves similar cases.**

> **Not for clinical use.** MediMemory is a research and demonstration project. It has not been validated for clinical use, has not been evaluated by clinicians, and has only been run on synthetic images produced by its own generator (`medical_data_generator.py`). The pathology "suggestions" come from a placeholder classifier with randomly initialised weights (see `PathologyClassifier` in `medimemory_server.py`), so they carry no diagnostic meaning.

## ✨ Features

- **🔬 Image Embeddings**: Feature extraction with an ImageNet-pretrained DenseNet-121 (no medical fine-tuning)
- **🧠 Pathology Suggestion Placeholder**: Multi-label output over 15 chest X-ray labels from randomly initialised weights; a stand-in for a trained model
- **📊 Similarity Search**: FAISS inner-product search over image embeddings, with text embeddings from sentence-transformers for reports
- **🔗 MCP Tools**: `store_medical_image`, `search_similar_cases`, `get_diagnosis_suggestions`
- **📈 Patient History**: Memories are stored per patient ID in SQLite

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Medical LLMs  │    │   MediMemory     │    │   Medical Data  │
│                 │    │   MCP Server     │    │                 │
│ • Radiology AI  │◄──►│                  │◄──►│ • Images (PNG)  │
│ • Clinical GPT  │    │ • Image Embeddings│    │ • Reports       │
│ • Diagnosis AI  │    │ • Pathology Class.│    │ • Patient Data  │
│ • Teaching LLMs │    │ • Similar Cases  │    │ • Metadata      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)
- Image files readable by Pillow (PNG/JPEG); DICOM is not supported

### Local Installation

```bash
git clone https://github.com/ln-bodicherla/MediMemory.git
cd MediMemory

# Install dependencies (no requirements file is included yet)
pip install torch torchvision faiss-cpu sentence-transformers mcp fastapi uvicorn pillow numpy pandas scikit-learn requests aiohttp python-multipart

# Generate synthetic sample data
python medical_data_generator.py

# Start the server (HTTP on port 8001)
python medimemory_server.py

# Run the demo client
python medimemory_client.py
```

### Docker Installation

The Dockerfile expects a `requirements-medimemory.txt` that is not in the repository yet; create it from the dependency list above before building.

```bash
# Build and run
docker build -f Dockerfile-medimemory -t medimemory .
docker run -p 8001:8001 -v $(pwd)/data:/app/data medimemory

# Or use docker-compose
docker-compose -f docker-compose-medimemory.yml up -d
```

## 📖 Usage

### Python Client SDK

```python
from medimemory_client import MediMemoryClient

# Initialize client
client = MediMemoryClient("http://localhost:8001")

# Upload medical image
memory_id = client.upload_medical_image(
    image_path="chest_xray.png",
    patient_id="PAT_12345", 
    modality="chest_xray",
    report_text="Chest X-ray shows possible consolidation in right lower lobe"
)

# Search similar cases
similar_cases = client.search_similar_cases(
    pathology="Pneumonia",
    limit=5
)

# Placeholder: the client currently returns hard-coded example values here
suggestions = client.get_diagnosis_suggestions("new_xray.png")
print(f"Top prediction: {suggestions['pathology_predictions']}")
```

Note: `MediMemoryClient.get_diagnosis_suggestions` does not call the server yet; it returns fixed demo values.

### MCP Tool Integration

The MCP tool handlers are defined in `medimemory_server.py`, but `main()` currently starts only the HTTP API and does not serve MCP over stdio, so this client configuration will not work until a stdio entry point is added.

```json
{
  "mcpServers": {
    "medimemory": {
      "command": "python",
      "args": ["/path/to/medimemory_server.py"],
      "env": {
        "MEDIMEMORY_DB_PATH": "/path/to/medical_data.db"
      }
    }
  }
}
```

## Evaluation

No benchmark or accuracy results are published. `medimemory_evaluation.py` contains an evaluation framework (Recall@K / Precision@K for retrieval, per-label F1 and AUC for classification, search time at different index sizes), but its `__main__` block only prints a message; you have to construct a `MediMemoryStore`, load data from `medical_data_generator.py`, and call `run_comprehensive_evaluation` yourself. Because the images are synthetic and the classifier weights are random, any numbers it produces only exercise the code paths.

## Possible Applications (not validated)

The following are directions this kind of system could be explored for in research settings. None of them have been tested.

### 1. Radiology Decision Support
- Retrieve similar cases for difficult diagnoses
- Compare current findings with historical outcomes
- Access teaching files for rare conditions

### 2. Longitudinal Patient Monitoring  
- Track disease progression over time
- Identify interval changes automatically
- Monitor treatment response

### 3. Medical Education
- Case-based learning with comprehensive examples
- Systematic exposure to pathology variations
- Interactive teaching tools for residents

### 4. Research and Population Health
- Identify patient cohorts with specific imaging characteristics
- Retrospective outcome analysis
- Clinical trial patient recruitment

## Security and Privacy

There are no HIPAA, encryption, access-control, audit-logging or de-identification features. Data is stored unencrypted in a local SQLite file and an in-memory FAISS index. Do not load real patient data.

## 🔧 Configuration

The server does not read environment variables yet: the database path defaults to `medimemory.db`, and the HTTP server listens on `0.0.0.0:8001` (set in `main()` in `medimemory_server.py`). The `MEDIMEMORY_*` variables in `docker-compose-medimemory.yml` are currently ignored.

## 🚀 Running

### Development
```bash
python medimemory_server.py
```

### Docker
```bash
docker-compose -f docker-compose-medimemory.yml up -d
```

## 📚 API Reference

### MCP Tools

#### `store_medical_image`
Store a medical image with its embeddings and placeholder pathology scores.

**Parameters:**
- `image_data` (string): Base64 encoded medical image
- `patient_id` (string): Patient identifier  
- `study_date` (string): Study date (ISO format)
- `modality` (string): Imaging modality
- `report_text` (string): Radiology report text
- `metadata` (object): Additional clinical metadata

#### `search_similar_cases`
Search for medically similar cases.

**Parameters:**
- `query_image` (string): Base64 encoded query image
- `query_text` (string): Text-based search query
- `patient_id` (string): Filter by specific patient
- `modality` (string): Filter by imaging modality
- `pathology` (string): Filter by pathology type
- `limit` (integer): Maximum results (default: 10)
- `threshold` (number): Similarity threshold (default: 0.7)

#### `get_diagnosis_suggestions`
Return placeholder pathology scores (random weights; not a diagnosis).

**Parameters:**
- `image_data` (string): Base64 encoded medical image
- `patient_history` (string): Patient medical history context

### HTTP Endpoints

- `POST /upload_medical_image` - Upload medical image
- `GET /search_cases` - Search medical cases
- `GET /health` - Health check

## 🐛 Troubleshooting

### Common Issues

**Server won't start:**
```bash
# Check Python version and dependencies
python --version  # Should be 3.11+
# (install the dependency list from Quick Start)

# Check available memory for ML models
free -h
```

**Image processing errors:**
```bash
# Verify image format support
python -c "from PIL import Image; print(Image.EXTENSION)"
```

**Memory/performance issues:**
```bash
# Monitor system resources
htop

# Check FAISS index size
ls -lh data/

# Enable GPU acceleration (if available)
# install the CUDA builds of PyTorch and FAISS for your platform
```

## 📄 License

No license file is included yet.

## 🙏 Acknowledgments

- **SuperMemory MCP Server** - Base memory architecture
- **FAISS** - High-performance vector similarity search
- **PyTorch** - Deep learning framework
- **Model Context Protocol** - Standardized LLM integration

---

*Disclaimer: This is a research and demonstration project. It is not a medical device, has not been validated for clinical use, and has only been run on synthetic images.*
