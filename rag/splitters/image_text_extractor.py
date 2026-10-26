import pytesseract
from pdf2image import convert_from_path

OCR_LANGUAGES = "eng+spa+fra"


def extract_text_from_image(image_path:str)->str:


  text=pytesseract.image_to_string(
    image_path,
    lang=OCR_LANGUAGES
  )

  return text



text=extract_text_from_image("./data/rag_image.webp")

with open("image_text.txt","w") as f:
  f.write(text)








