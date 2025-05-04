# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import fitz  # PyMuPDF
import numpy as np
import cv2
import pytesseract
from PIL import Image
import logging
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from groq import Groq
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Load TrOCR model for handwritten text
logger.info("Loading TrOCR model for handwritten text recognition...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
logger.info("TrOCR model loaded successfully")

def is_handwritten(image):
    """
    Determine if an image likely contains handwritten text
    Simple heuristic based on contour analysis
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyze contour properties characteristic of handwriting
    if len(contours) < 5:
        return False
    
    # Check for irregular contours typical in handwriting
    irregular_shapes = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20:  # Skip very small contours
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.2:  # Handwritten strokes often have low circularity
                irregular_shapes += 1
    
    return irregular_shapes > len(contours) * 0.3  # If over 30% are irregular

def extract_text_from_handwritten(image):
    """
    Extract text from handwritten content using TrOCR
    """
    try:
        # Convert OpenCV image to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Process image with TrOCR
        pixel_values = processor(pil_image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text
    except Exception as e:
        logger.error(f"Error extracting handwritten text: {str(e)}")
        return ""

def extract_text_from_printed(image):
    """
    Extract text from printed content using Tesseract OCR
    """
    try:
        # Apply preprocessing to improve OCR results
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Apply OCR
        text = pytesseract.image_to_string(thresh)
        return text
    except Exception as e:
        logger.error(f"Error extracting printed text: {str(e)}")
        return ""

def process_pdf_page(page):
    """
    Process a single PDF page for text extraction
    """
    try:
        # Render page to an image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase resolution for better OCR
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        np_img = np.array(img)
        
        # Convert RGB to BGR for OpenCV processing
        cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        
        # Determine if the page contains handwritten text
        if is_handwritten(cv_img):
            logger.info("Detected handwritten text, using TrOCR")
            return extract_text_from_handwritten(cv_img)
        else:
            logger.info("Detected printed text, using Tesseract")
            return extract_text_from_printed(cv_img)
    except Exception as e:
        logger.error(f"Error processing PDF page: {str(e)}")
        return ""

def summarize_and_generate_questions(text: str) -> dict:
    prompt = f"""
You are an AI academic assistant.

Given the following text, perform two tasks:

1. Provide a clear, concise summary of the text.
2. Generate 5 important exam-style questions along with their answers.

Format your response in the following JSON structure:
{{
  "summary": "...",
  "qa_pairs": [
    {{
      "question": "...",
      "answer": "..."
    }},
    ...
  ]
}}

Text:
{text}
"""

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # Or llama3-8b-8192 depending on speed/quality needs
            messages=[
                {"role": "system", "content": "You are a helpful study assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )

        content = response.choices[0].message.content

        # Try to convert the string response to a dictionary
        import json
        import re

        # Extract JSON from response using regex
        json_str = re.search(r'\{.*\}', content, re.DOTALL)
        if json_str:
            return json.loads(json_str.group())
        else:
            return {"error": "Failed to extract JSON from model response."}

    except Exception as e:
        return {"error": str(e)}
    

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF, handling both printed and handwritten text
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        
        for page_num in range(len(doc)):
            logger.info(f"Processing page {page_num+1}/{len(doc)}")
            page = doc.load_page(page_num)
            
            # First try to extract text directly if it's a digital PDF
            text = page.get_text()
            
            # If no text is found, the PDF might be scanned or contain images
            if not text.strip():
                text = process_pdf_page(page)
            
            full_text.append(text)
        
        doc.close()
        return "\n\n--- PAGE BREAK ---\n\n".join(full_text)
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return f"Error extracting text: {str(e)}"

@app.route('/summarize', methods=['POST'])
def extract_text():
    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400
    
    results = []
    
    for file in files:
        try:
            # Create a unique filename
            unique_filename = str(uuid.uuid4()) + '.pdf'
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            
            # Save the PDF temporarily
            file.save(file_path)
            logger.info(f"Processing file: {file.filename} (saved as {unique_filename})")
            
            # Extract text from the PDF
            extracted_text = extract_text_from_pdf(file_path)
            
            # Add to results
            results.append({
                "filename": file.filename,
                "text": extracted_text
            })
            
            # Clean up
            try:

                # print(f"Removing temporary file {file_path}")
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    # print(f"Results: {results}")
    alltext = ""
    for result in results:
        alltext += result.get('text', '') + "\n"
    res = summarize_and_generate_questions(alltext)
    print(f"Summary and questions: {res}")
    return jsonify({"results": res, "pdf_content": alltext})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Make sure Tesseract is installed and configured
    try:
        pytesseract.get_tesseract_version()
        logger.info("Tesseract OCR is properly installed")
    except Exception as e:
        logger.error(f"Tesseract OCR is not properly installed: {str(e)}")
    
    logger.info("Starting PDF OCR server...")
    app.run(host='0.0.0.0', port=5000, debug=True)