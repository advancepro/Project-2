# import dp as dp
#import tokenizer as tokenizer
import torchtext as torchtext
from nltk import ngrams
from torch.utils.data import DataLoader


# torchtext.data.utils.ngrams_iterator(ngrams)
# torchtext.data.utils.get_tokenizer(tokenizer, language='en')

# from torch.utils.data.backward_compatibility import worker_init_fn
# DataLoader(dp, num_workers=4, worker_init_fn=worker_init_fn, drop_last=True)

print("\033[031;40m The William Randolph Hearst Foundation \n"
      "will give $1.25 million \033[032;40m to Lincoln Center, Metropolitan Opera Co., \n"
      "\033[034;40m New York Philharmonic and Juilliard School. \n"
      "“Our board felt that we had a \n"
      "\033[035;40m real opportunity to make a mark on the future of the performing arts \n")

print("\033[031;40m Arts \033[032;40m Budgets \033[034;40m Children \033[035;40m Education \n")
print("-------------------------------------------------------------------------- \n "
      "\033[037;40m"
      "NEW MILLION CHILDREN SCHOOL \n"
      "FILM TAX WOMEN STUDENTS \n"
      "SHOW PROGRAM PEOPLE SCHOOLS \n"
      "MUSIC BUDGET CHILD EDUCATION \n"
      "MOVIE BILLION YEARS TEACHERS \n"
      "PLAY FEDERAL FAMILIES HIGH\n "
      "MUSICAL YEAR WORK PUBLIC\n "
      "BEST SPENDING PARENTS TEACHER \n"
      "ACTOR NEW SAYS BENNETT \n"
      "FIRST STATE FAMILY MANIGAT \n"
      "YORK PLAN WELFARE NAMPHY \n"
      "OPERA MONEY MEN STATE \n"
      "THEATER PROGRAMS PERCENT PRESIDENT \n"
      "ACTRESS GOVERNMENT CARE ELEMENTARY \n"
      "LOVE CONGRESS LIFE HAITI \n")

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

def summarize(text, per):
    nlp = spacy.load('en_core_web_sm')
    doc= nlp(text)
    tokens=[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    return summary


import torch
from torch import nn
from torch import optim
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings("ignore")

transform = transforms.Compose([
      transforms.Resize((32, 32)),
      transforms.ToTensor()
])

flat_img = 3072  

img = Image.open('model.PNG')
real_img = transform(img)

torch.manual_seed(2)
fake_img = torch.rand(1, 100)

plt.imshow(np.transpose(real_img.numpy(), (1, 2, 0)))
print(real_img.size())

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Sequential(
        nn.Linear(flat_img, 10000),
        nn.ReLU(),
        nn.Linear(10000, 1),
        nn.Sigmoid()
    )

  def forward(self, img):
    img = img.view(1, -1)
    out = self.linear(img)

    return out

class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Sequential(
        nn.Linear(100, 10000),
        nn.LeakyReLU(),
        nn.Linear(10000, 4000),
        nn.LeakyReLU(),
        nn.Linear(4000, flat_img)
    )

  def forward(self, latent_space):
    latent_space = latent_space.view(1, -1)
    out = self.linear(latent_space)

    return out

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

discr = Discriminator().to(device)
gen = Generator().to(device)

opt_d = optim.SGD(discr.parameters(), lr=0.001, momentum=0.9)
opt_g = optim.SGD(gen.parameters(), lr=0.001, momentum=0.9)

criterion = nn.BCELoss()

epochs = 500
discr_e = 4
gen_e = 3

# whole model training starts here
for epoch in tqdm(range(epochs), total=epochs):

      # discriminator training
      for k in range(discr_e):
            opt_d.zero_grad()

            out_d1 = discr(real_img.to(device))
            # loss for real image
            loss_d1 = criterion(out_d1, torch.ones((1, 1)).to(device))
            loss_d1.backward()

            out_d2 = gen(fake_img.to(device)).detach()
            # loss for fake image
            loss_d2 = criterion(discr(out_d2.to(device)), torch.zeros((1, 1)).to(device))
            loss_d2.backward()

            opt_d.step()

      # generator training
      for i in range(gen_e):
            opt_g.zero_grad()

            out_g = gen(fake_img.to(device))

            # Binary cross entropy loss
            # loss_g =  criterion(discr(out_g.to(device)), torch.ones(1, 1).to(device))

            # ----Loss function in the GAN paper
            # [log(1 - D(G(z)))]
            loss_g = torch.log(1.0 - (discr(out_g.to(device))))
            loss_g.backward()

            opt_g.step()

      plt.figure(figsize=(8, 8))
      plt.subplot(1, 2, 1)
      plt.title("Generated Image")
      plt.xticks([])
      plt.yticks([])
      plt.imshow(np.transpose(out_g.resize(3, 32, 32).cpu().detach().numpy(), (1, 2, 0)))

      plt.subplot(1, 2, 2)
      plt.title("Original Image")
      plt.xticks([])
      plt.yticks([])
      plt.imshow(np.transpose(real_img.numpy(), (1, 2, 0)))
      plt.show()