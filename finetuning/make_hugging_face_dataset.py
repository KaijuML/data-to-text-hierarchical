from datasets import load_dataset

def parse_hugging_face_dataset():
    hf_data = load_dataset(GEM/sportsett_basketball)
    with open('../data/hugging_face/train_input.txt', 'w') as f:
        f.write(hf_data)