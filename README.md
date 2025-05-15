# Simple CLIP Search with openai/CLIP
This is the openai's CLIP version of Simple CLIP Search. This implementation is based on
[SCS](https://github.com/matsui528/scs).

<img width="588" alt="Screenshot 2025-05-15 at 18 33 57" src="https://github.com/user-attachments/assets/f15e5828-6994-45b8-97d1-3b237f342ec9" />

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
