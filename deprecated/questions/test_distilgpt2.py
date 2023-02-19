from datasets import load_dataset

if __name__ == '__main__':
    data_path = './apps_flyers.json'
    dataset = load_dataset("json", data_files=f'.{data_path}', split='train')

    eli5 = load_dataset("eli5", split="train_asks[:5000]")
    print(1)