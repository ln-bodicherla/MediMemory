"""
Medical Dataset Generator for MediMemory Evaluation
Generates synthetic medical imaging data with realistic characteristics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import uuid
import random
from PIL import Image, ImageDraw, ImageFilter
import io
import base64

class MedicalDataGenerator:
    """Generate synthetic medical imaging data for evaluation"""

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        self.pathologies = [
            "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", 
            "Infiltration", "Mass", "Nodule", "Pneumonia", 
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", 
            "Fibrosis", "Pleural_Thickening", "Hernia"
        ]

        self.modalities = ["chest_xray", "ct_chest", "mri_brain"]
        self.hospitals = ["General Hospital", "Medical Center", "University Hospital"]

        # Pathology co-occurrence patterns (simplified)
        self.pathology_correlations = {
            "Pneumonia": ["Consolidation", "Infiltration"],
            "Edema": ["Cardiomegaly", "Effusion"],
            "Mass": ["Nodule"],
            "Atelectasis": ["Effusion"]
        }

    def generate_synthetic_xray(self, pathologies, image_size=(224, 224)):
        """Generate synthetic chest X-ray image with pathological patterns"""
        # Create base lung field
        img = Image.new('L', image_size, color=40)  # Dark background
        draw = ImageDraw.Draw(img)

        # Draw lung fields
        lung_left = [(20, 50), (90, 50), (100, 180), (30, 190)]
        lung_right = [(130, 50), (200, 50), (190, 190), (120, 180)]

        draw.polygon(lung_left, fill=120, outline=100)
        draw.polygon(lung_right, fill=120, outline=100)

        # Add pathological patterns based on labels
        if "Cardiomegaly" in pathologies:
            # Enlarged heart shadow
            draw.ellipse([80, 120, 140, 200], fill=60, outline=50)

        if "Pneumonia" in pathologies:
            # Consolidation pattern
            for _ in range(5):
                x = random.randint(30, 90)
                y = random.randint(80, 150)
                draw.ellipse([x-10, y-10, x+10, y+10], fill=80)

        if "Effusion" in pathologies:
            # Pleural effusion (fluid collection)
            draw.polygon([(20, 170), (100, 170), (100, 190), (30, 190)], fill=90)

        if "Mass" in pathologies:
            # Focal mass
            x = random.randint(40, 180)
            y = random.randint(60, 140)
            draw.ellipse([x-15, y-15, x+15, y+15], fill=70, outline=60)

        if "Pneumothorax" in pathologies:
            # Air collection pattern
            for _ in range(3):
                x = random.randint(130, 190)
                y = random.randint(60, 120)
                draw.ellipse([x-8, y-8, x+8, y+8], fill=140)

        # Add noise and blur for realism
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        # Convert to numpy array and add noise
        img_array = np.array(img)
        noise = np.random.normal(0, 5, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def generate_radiology_report(self, pathologies, patient_age, sex):
        """Generate realistic radiology report text"""
        reports = {
            "No Finding": "Clear lung fields. No acute cardiopulmonary abnormality.",
            "Pneumonia": "Patchy consolidation in the right lower lobe consistent with pneumonia.",
            "Cardiomegaly": "Enlarged cardiac silhouette suggesting cardiomegaly.",
            "Effusion": "Small to moderate pleural effusion noted on the right side.",
            "Mass": "Focal opacity in the left upper lobe concerning for mass lesion.",
            "Atelectasis": "Linear atelectasis in the left lower lobe.",
            "Pneumothorax": "Small pneumothorax on the right side.",
            "Edema": "Bilateral pulmonary edema with vascular congestion.",
            "Infiltration": "Bilateral lower lobe infiltrates consistent with infection.",
            "Consolidation": "Dense consolidation in the right middle lobe."
        }

        if not pathologies or "No Finding" in pathologies:
            return "IMPRESSION: No acute cardiopulmonary abnormality identified."

        findings = []
        for pathology in pathologies:
            if pathology in reports:
                findings.append(reports[pathology])

        report = f"CLINICAL HISTORY: {patient_age}-year-old {sex.lower()} patient.\n\n"
        report += "FINDINGS:\n"
        report += "\n".join(findings)
        report += "\n\nIMPRESSION: " + "; ".join([p.replace("_", " ") for p in pathologies]) + "."

        return report

    def generate_patient_dataset(self, num_patients=100, studies_per_patient=3):
        """Generate complete synthetic dataset"""
        patients = []
        studies = []

        for i in range(num_patients):
            patient_id = f"PAT_{i:05d}"
            age = random.randint(18, 90)
            sex = random.choice(["M", "F"])

            patients.append({
                "patient_id": patient_id,
                "age": age,
                "sex": sex,
                "hospital": random.choice(self.hospitals)
            })

            # Generate multiple studies for this patient
            base_date = datetime.now() - timedelta(days=random.randint(1, 365))

            for j in range(studies_per_patient):
                study_date = base_date + timedelta(days=j * random.randint(30, 120))

                # Simulate disease progression/development
                if j == 0:
                    # Initial study - higher chance of no finding
                    if random.random() < 0.4:
                        pathologies = ["No Finding"]
                    else:
                        primary = random.choice(self.pathologies[1:])  # Exclude "No Finding"
                        pathologies = [primary]

                        # Add correlated pathologies
                        if primary in self.pathology_correlations:
                            for corr_path in self.pathology_correlations[primary]:
                                if random.random() < 0.3:
                                    pathologies.append(corr_path)
                else:
                    # Follow-up studies - progression or improvement
                    prev_study = [s for s in studies if s["patient_id"] == patient_id][-1]
                    prev_pathologies = prev_study["pathologies"]

                    if "No Finding" in prev_pathologies:
                        # New condition may develop
                        if random.random() < 0.3:
                            pathologies = [random.choice(self.pathologies[1:3])]
                        else:
                            pathologies = ["No Finding"]
                    else:
                        # Existing condition progression
                        if random.random() < 0.6:
                            pathologies = prev_pathologies.copy()
                            # May add complications
                            if random.random() < 0.2:
                                pathologies.append(random.choice(self.pathologies[1:]))
                        else:
                            # Improvement
                            if len(prev_pathologies) > 1:
                                pathologies = prev_pathologies[:1]
                            else:
                                pathologies = ["No Finding"] if random.random() < 0.5 else prev_pathologies

                # Generate synthetic image
                modality = "chest_xray"  # Focus on chest X-rays for this demo
                image = self.generate_synthetic_xray(pathologies)

                # Convert image to base64
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode()

                # Generate report
                report = self.generate_radiology_report(pathologies, age, sex)

                studies.append({
                    "study_id": f"STU_{len(studies):05d}",
                    "patient_id": patient_id,
                    "study_date": study_date,
                    "modality": modality,
                    "pathologies": pathologies,
                    "report_text": report,
                    "image_base64": image_base64,
                    "metadata": {
                        "age": age,
                        "sex": sex,
                        "hospital": patients[-1]["hospital"],
                        "study_number": j + 1
                    }
                })

        return pd.DataFrame(patients), pd.DataFrame(studies)

# Generate sample dataset for testing
if __name__ == "__main__":
    print("🔬 Generating synthetic medical dataset...")
    generator = MedicalDataGenerator()
    patients_df, studies_df = generator.generate_patient_dataset(num_patients=50, studies_per_patient=2)

    print(f"✅ Generated {len(patients_df)} patients and {len(studies_df)} studies")
    print(f"📊 Pathology distribution:")
    pathology_counts = {}
    for pathologies in studies_df['pathologies']:
        for p in pathologies:
            pathology_counts[p] = pathology_counts.get(p, 0) + 1

    for path, count in sorted(pathology_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {path}: {count}")

    # Save sample data
    patients_df.to_csv('sample_patients.csv', index=False)
    studies_df.to_csv('sample_studies.csv', index=False)
    print("💾 Saved sample data to CSV files")
