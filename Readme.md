# Meme Text Extractor

This short python script uses OpenCV to prepare an image for tesseract to perform OCR on.  
This script assumes the meme style that is to say, an image with large block capital letters on.

## Configuration

Tesseract configuration is given in config.json file.

## Dependencies

* Requires the Tesseract binary
* Python modules:

```
pip install opencv-python
pip install pytesseract
pip install matplotlib
```

## Usage

```
./meme-extractor.py [options] <image> 
```

The output will be the text from the image with \<br/\> between the lines.

### Options

* --debug output debugging images


## Reference

Original concept code was from [here](https://towardsdatascience.com/extract-text-from-memes-with-python-opencv-tesseract-ocr-63c2ccd72b69).


