# Project-2

Programming Language: Python
Method: Text Classification

My aim of the project is to implement the paper information with the help of Classification which is a branch of Data Science. 

There are some situations where the article has too much information and needs to be summarized and the goal is to minimize the time you spend reading. 

So the classification helps to print out the short information after downloading and parsing the information with the help of summarization.

The Book used:

Latent Dirichlet Allocation (For iteration)

Summarization beyond sentence extraction: A probabilistic approach to sentence compression (For summarization)

Libraries:

* TorchText.data.utils
* torchtext.data.utils.get_tokenizer(tokenizer, language=’en’)
* import torchtext

The steps were

- article = Article(url)
- article.download()
- article.parse()
- article.text

