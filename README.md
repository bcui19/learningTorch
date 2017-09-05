# learningTorch
trying to learn torch because it seems really elegant and sometimes tensorflow is not the most fun thing to work with

Most of these modules are brought down from:

http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

I generally put random comments where I think something is interesting in the moment, a lot of code isn't super modularized, but it can be easily made so 

Also I don't think a lot of these are minibatched properly, so minibatch size is 1; I also think this isn't too hard to fix

## Comments on RNN_Generator
For this I followed the architecture presented in the pytorch tutorial. After training for 50,000 samples (no minibatching) and SGD I was able to achieve the following reasonable result:

'Alov'

However, it must be noted that this took many iterations of refreshing the network from scratch -- many times the network would output something too simple to be considered reasonable. I haven't thought/derived a lot of the work, but I think this is mainly due to vanishing gradients are in the architecture presented. I think how the architecture is designed it acts very similarly to a vanilla RNN cell, which is well known to have vanishing gradient issues.

