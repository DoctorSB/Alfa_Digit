from torch.utils.data import Dataset

index_to_tag = ["O", "B-GOOD", "I-GOOD", "B-BRAND", "I-BRAND", "PAD"]
tag_to_index = {tag: index for index, tag in enumerate(index_to_tag)}

class ReceiptsDataset(Dataset):
    def __init__(self, df, fasttext):
        super().__init__()
        self.is_predict = "tags" not in df.columns
        self.data = df[["tokens", "good", "brand", "tags"]] if not self.is_predict else df[["tokens", "id"]]
        self.data = self.data.values
        self.fasttext = fasttext

    def __getitem__(self, index):
        identifier = 0 if not self.is_predict else self.data[index][1]
        tokens = self.data[index][0]
        embeddings = self.fasttext.wv[tokens]
        goods = self.data[index][1].split(',') if not self.is_predict else list()
        brands = self.data[index][2].split(',') if not self.is_predict else list()
        tags = self.data[index][3] if not self.is_predict else ["O"] * len(tokens)
        target = [tag_to_index[tag] for tag in tags]
        return identifier, tokens, embeddings, goods, brands, target

    def __len__(self):
        return len(self.data)