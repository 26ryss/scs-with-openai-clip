# Simple CLIP Search with openai/CLIP
This is the openai's CLIP version of Simple CLIP Search. This implementation is based on
[SCS](https://github.com/matsui528/scs).

## Setup
Please refer to the "Usage" section in [CLIP](https://github.com/openai/CLIP) to setup for using CLIP.

Also, install pillow.
```bash
pip install pillow
```
## Usage
Please generate image features by runnning
```
python extract_features.py
```

Once you create the image features, you can search images based on a query.
```
python search.py
```