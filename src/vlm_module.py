from pydantic import BaseModel, Field
from ollama import Client

class Candidate(BaseModel):
    number: int = Field(description="The candidate number (หมายเลขผู้สมัคร)")
    name: str = Field(description="The full name of the candidate (ชื่อผู้สมัคร)")
    party: str = Field(description="The political party of the candidate (สังกัดพรรคการเมือง)")
    score: int = Field(description="The handwritten score or votes received (คะแนนที่ได้)")

class ElectionForm(BaseModel):
    candidates: list[Candidate] = Field(description="List of all candidates in the table")
    total_valid_votes: int = Field(description="Total valid votes if written at the bottom")

ollama_client = Client(host='http://localhost:11434')
MODEL_NAME = 'scb10x/typhoon-ocr1.5-3b'

def extract_full_table(image_crop_path):
    """
    Typhoon OCR 1.5 execution via Ollama for Full Table Extraction
    """
    try:
        response = ollama_client.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': 'You are a highly accurate OCR system. Extract all candidate data from this Thai election form table. Make sure to capture every row accurately, including candidate number, name, party, and the handwritten score.',
                'images': [image_crop_path]
            }],
            format=ElectionForm.model_json_schema(),
            options={
                'temperature': 0.0
            }
        )
        
        extracted_data = ElectionForm.model_validate_json(response.message.content)
        return extracted_data
    except Exception as e:
        print(f"Error extracting table: {e}")
        return None
