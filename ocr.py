from transformers import AutoProcessor, AutoModelForCausalLM  
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import re
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

def ocr_tesseract(image):
    return pytesseract.image_to_string(image, lang="rus+eng").strip()

def preprocess_image(path):
    img = Image.open(path).convert("L")  # grayscale
    img = img.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    return img.convert("RGB")

model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
task_prompt = "<OCR>"
# MODEL = "Akajackson/donut_rus"
# processor = DonutProcessor.from_pretrained(MODEL)
# model = VisionEncoderDecoderModel.from_pretrained(MODEL)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)  # doctest: +IGNORE_RESULT
# task_prompt = "<cord-v2>"

# ocr = PaddleOCR(use_angle_cls=True, lang='ru', det_db_box_thresh=0.3)

def parse_text_from_image(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer["<OCR>"]

    # decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    # pixel_values = processor(image, return_tensors="pt").pixel_values
    # outputs = model.generate(
    #     pixel_values.to(device),
    #     decoder_input_ids=decoder_input_ids.to(device),
    #     max_length=model.decoder.config.max_position_embeddings,
    #     pad_token_id=processor.tokenizer.pad_token_id,
    #     eos_token_id=processor.tokenizer.eos_token_id,
    #     use_cache=True,
    #     bad_words_ids=[[processor.tokenizer.unk_token_id]],
    #     return_dict_in_generate=True,
    # )
    # sequence = processor.batch_decode(outputs.sequences)[0]
    # sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    # sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    # print(processor.token2json(sequence))
    # return processor.token2json(sequence)
    
    
# def recognize_text(image_path):
#     result = ocr.ocr(image_path, cls=True)
#     if not result or not result[0]:
#         return "❌ Текст не найден."
    
#     lines = [line[1][0] for line in result[0]]
#     return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    image_path = "/hdd/zhuldyzzhan/aigerim/Screenshot 2025-04-17 at 21.32.39.png"
    image_pil = Image.open(image_path).convert("RGB")  # Replace with your image file path
    text = parse_text_from_image(task_prompt, image_pil)
    # text = recognize_text(image_path)
    print(f"Parsed Text: {text}")
    
    # enhanced_image = preprocess_image(image_path)
    # text = ocr_tesseract(enhanced_image)
    # print(f"Parsed Text: {text}")