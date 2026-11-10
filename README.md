# text-eraser-test

Test page for a prototype that finds text in the camera image and fills it in from the surrounding background (no OCR, no reading).

Open: https://tsumikiroom.github.io/text-eraser-test/

- Text detector: PaddleOCR PP-OCRv5 mobile det (ONNX, Apache-2.0), from Hugging Face webnn/PP-OCRv5-ONNX (default)
- Older detector for comparison (?model=v4): PaddleOCR PP-OCRv4 det (ONNX, Apache-2.0), from Hugging Face SWHL/RapidOCR
- Runtime: onnxruntime-web 1.20.1 (loaded from jsDelivr)

Temporary test repository.
