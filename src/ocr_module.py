from paddleocr import PaddleOCR

# Initialize PaddleOCR strictly for the Thai printed text cells
try:
    ocr_printed = PaddleOCR(use_angle_cls=True, lang='thai', show_log=False)
except Exception as e:
    print(f"Failed to load PaddleOCR: {e}")
    ocr_printed = None

def process_printed_text(image_crop):
    """
    PaddleOCR execution for Candidate and Party names
    """
    if ocr_printed is None:
        return "", 0.0
        
    try:
        result = ocr_printed.ocr(image_crop, cls=True)
        if not result or not result[0]:
            return "", 0.0
            
        # Extract the decoded text string and the float confidence score
        # PaddleOCR returns list of [[box, (text, confidence)], ...]
        text_parts = []
        conf_sum = 0.0
        
        for line in result[0]:
            text, conf = line[1]
            text_parts.append(text)
            conf_sum += conf
            
        final_text = " ".join(text_parts)
        avg_conf = conf_sum / len(result[0])
        
        return final_text, avg_conf
    except Exception as e:
        print(f"OCR Error: {e}")
        return "", 0.0
