import torch
import clip
from PIL import Image
import torch.nn as nn
import torch
import collections
import json
import os
import torchvision.transforms as transforms
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import numpy as np


def load_image(file_name, preprocess):
    file_path = 'data/coco/images'
    dic = collections.OrderedDict()
    for i, image in enumerate(file_name):
        img_path = '%s/%s.jpg' % (file_path, image)
        new_img = Image.open(img_path)
        new_img = preprocess(new_img)
        dic[image] = new_img
    # print(dic)
    # img_set = torch.Tensor(img_set)
    return dic


class Clip_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        self.image_entries = []
        self.data = []
        self.all_data = []
        data_root_2 = 'data/image_retrieval/'
        annotations_jsonpath = os.path.join(data_root_2, "dataset_coco_test_final.json")
        self.all_data.extend(json.load(open(annotations_jsonpath)))
        self.total_size = 1000
        self.all_data = self.all_data[:self.total_size * 5]  # 加载全部的sent
        for annotation in self.all_data:
            image_id = annotation['filename'][:-4]
            if image_id not in self.image_entries:
                self.image_entries.append(image_id)
            self.data.append({"caption": annotation['sentence'], "image_id": image_id})

        self.dic = load_image(self.image_entries, self.preprocess)

    def sent_retrieval(self, image, text):
        # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
        # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
        # file_path = 'data/coco/images'
        # img_path = '%s/%s.jpg' % (file_path, image[0][0])
        image_preprocessed = self.dic[image[0]].unsqueeze(0).to(self.device)

        text = clip.tokenize(t[0] for t in text).to(self.device)

        logits_per_image, logits_per_text = self.model(image_preprocessed, text)

        return logits_per_image, logits_per_text

    def forward(self, image, text):
        # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
        # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
        # file_path = 'data/coco/images'
        # img_path = '%s/%s.jpg' % (file_path, image[0][0])
        image_preprocessed = self.dic[image[0][0]].unsqueeze(0).to(self.device)
        for im_data in image[1:]:
            temp = self.dic[im_data[0]].unsqueeze(0).to(self.device)
            image_preprocessed = torch.cat((image_preprocessed, temp), 0)
        text = clip.tokenize(text).to(self.device)

        logits_per_image, logits_per_text = self.model(image_preprocessed, text)

        return logits_per_image, logits_per_text


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN101", device=device)
    image_input = Image.open("CAT.jpg")
    # re_img = transforms.Resize(64)(image_input)
    image = preprocess(image_input).to(device)
    real_imgs = image.cpu().data.numpy()
    # c x h x w --> h x w x c
    real_imgs = (np.transpose(real_imgs, (1, 2, 0)) + 2) / 4.0 * 255.0
    PIL_im = Image.fromarray(np.uint8(real_imgs))
    PIL_im.save('clip.png')
    text = clip.tokenize(["a diagram and a dog", "a dog", "a cat"]).to(device)
    # tokenize = _tokenizer.decode(text[0][1:3].cpu().numpy())
    # print(tokenize)
    sentence = ''
    tmp = text[0].data.cpu().numpy()
    for i in tmp[1:2 + 1]:
        print(i)
        word = _tokenizer.decoder[i].replace('</w>', ' ')
        # word = _tokenizer.byte_decoder[word].decode('utf-8', errors="replace").replace('</w>', ' ')
        sentence += word
    # sentence = bytearray([_tokenizer.byte_decoder[c] for c in sentence]).decode('utf-8', errors="replace").replace('</w>', ' ')
    print(sentence)
    with torch.no_grad():
        img = image.unsqueeze(0)
        print(img.size())
        image_features, cnn_code = model.encode_image(image.unsqueeze(0))
        print(image_features.size(), cnn_code.size())
        text_features, word_embs = model.encode_text(text)
        print(text_features.size(), word_embs.size())
        logits_per_image, logits_per_text = model(image.unsqueeze(0), text)
        print(logits_per_image, logits_per_text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print(probs)
