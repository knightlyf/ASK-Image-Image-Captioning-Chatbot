# ASK-Image-Image-Captioning-Chatbot
“Ask Image” is a system which automatically generates natural language descriptions based on the image provided
Caption generation is a challenging artificial intelligence problem where a textual description must be
generated for a given photograph. It requires both methods from computer vision to understand the
content of the image and a language model from the field of natural language processing to turn the
understanding of the image into words in the right order. A single end-to-end model can be defined to
predict a caption, given a photo, instead of requiring sophisticated data preparation or a pipeline of
specifically designed models. Conversational AI use cases are diverse. They include customer support,
e-commerce, controlling IoT devices, enterprise, productivity and much more. In very simplistic terms,
these use cases involve a user asking a specific question (intent) and the conversational experience (or
the chatbot) responding to the question by making calls to a backend system like a CRM, Database or
an API. It turns out that some of these use cases can be enriched by allowing a user to upload an image.
In such cases, you would want the conversation experience to take an action based on what exactly is in
that image. In this project we will develop a photo captioning deep learning model and integrate
COCOS dataset, SQuAD dataset to provide rich and dynamic ML based responses to user provided
image inputs.

Proposed Architecture

 Image Pre-processing: Pre-processing is a common name for operations with images at the

lowest level of abstraction - both input and output are intensity images. The aim of pre-
processing is an improvement of the image data that suppresses unwanted distortions or

enhances some image features important for further processing.
 Image Encoding: It is also known as “encoding method”. Image encoding is used to prepare
photos to be displayed in a way that most computers and software, as well as browsers, can
support. This is often necessary because not all image viewing software is able to open certain
image files.



 RNN Classification: Recurrent Neural Networks Recurrent Neural Networks (RNN) are a
type of Neural Network where the output from the previous step is fed as input to the current
step.

Chatbot Architecture:


 Information Retrieval-based QA: IR-based QA systems are sometimes called text-based and
relies on unstructured corpus - huge amount of paragraphs on web sites such as news sites or
Wikipedia. As can be seen from its name IR methods are used in order to extract passages that
can contain an answer to given question. The key phrases or keywords from a question which
determine answer type make search query for search engine. The search engine returns the
documents which are splitted into many passages. The final possible answer strings are chosen
from those passages and most fitted answer is selected as a result. Most of modern open-domain
QA systems are IR-based.
 Question Processing: After question processing, some important data is extracted from a
question. Based on this information our task is to determine type of answer. It is called answer
type recognition, or just question classification and rely on name entity recognition in most of
the cases.
 Document Processing: The next step is formulation of queries. For that we use query
reformulation rules. The construted query is sent to information retrieval engine running based
on a number of indexed documents. As a result, we get a set of documents which are ranked by
relevance. The next step is retrieving units – passages, sentences or sections from a large set of

13

documents. First we filter documents which do not contain the entities we got from answer type
recognition phase. Secondly, we filter and rank other documents with using simple machine
learning.
 Answer Processing: The last step is an extraction of answer for a question from the selected
passage or sentence. The main part of this work will focus on an answer processing. I will
analyze different advanced sequential models based on embeddings of the given question and the
selected passage.
 Language Dataset: Here we have used SQuAD dataset. Stanford Question Answering Dataset
(SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on
a set of Wikipedia articles, where the answer to every question is a segment of text, or span,
from the corresponding reading passage, or the question might be unanswerable.

##Implementation Details
Algorithms/ Techniques:
Convolutional Neural Network (CNN):
CNN image classifications takes an input image, process it and classify it under certain categories (Eg.,
Dog, Cat, Tiger, Lion). Computers sees an input image as array of pixels and it depends on the image
resolution. Based on the image resolution, it will see h x w x d( h = Height, w = Width, d = Dimension ).
Eg., An image of 6 x 6 x 3 array of matrix of RGB (3 refers to RGB values) and an image of 4 x 4 x 1
array of matrix of grayscale image. Technically, deep learning CNN models to train and test, each input
image will pass it through a series of convolution layers with filters (Kernals), Pooling, fully connected
layers (FC) and apply Softmax function to classify an object with probabilistic values between 0 and 1.
Convolution of an image with different filters can perform operations such as edge detection, blur and
sharpen by applying filters.
Inception v3:
Inception v3 is a widely-used image recognition model that has been shown to attain greater than 78.1%
accuracy on the ImageNet dataset. The model is the culmination of many ideas developed by multiple
researchers over the years. It is based on the original paper: "Rethinking the Inception Architecture for
Computer Vision" by Szegedy, et. al.
The model itself is made up of symmetric and asymmetric building blocks, including convolutions,
average pooling, max pooling, concats, dropouts, and fully connected layers. Batchnorm is used
extensively throughout the model and applied to activation inputs. Loss is computed via Softmax.



Recurrent Neural Network (RNN):
Recurrent Neural Network is a generalization of feedforward neural network that has an internal
memory. RNN is recurrent in nature as it performs the same function for every input of data while the
output of the current input depends on the past one computation. After producing the output, it is copied
and sent back into the recurrent network. For making a decision, it considers the current input and the
output that it has learned from the previous input. Unlike feedforward neural networks, RNNs can use
their internal state (memory) to process sequences of inputs. This makes them applicable to tasks such
as unsegmented, connected handwriting recognition or speech recognition. In other neural networks, all
the inputs are independent of each other. But in RNN, all the inputs are related to each other.

Bahdanau Attention:
Bahdanau et al. proposed an attention mechanism that learns to align and translate jointly. It is also
known as Additive attention as it performs a linear combination of encoder states and the decoder states.



BERT (Bidirectional Encoder Representations from Transformers):
BERT (Bidirectional Encoder Representations from Transformers) is a recent paper published by
researchers at Google AI Language. It has caused a stir in the Machine Learning community by
presenting state-of-the-art results in a wide variety of NLP tasks, including Question Answering (SQuAD
v1.1), Natural Language Inference (MNLI), and others.

BERT’s key technical innovation is applying the bidirectional training of Transformer, a popular
attention model, to language modelling. This is in contrast to previous efforts which looked at a text
sequence either from left to right or combined left-to-right and right-to-left training. The paper’s results
show that a language model which is bidirectionally trained can have a deeper sense of language context
and flow than single-direction language models. In the paper, the researchers detail a novel technique
named Masked LM (MLM) which allows bidirectional training in models in which it was previously
impossible.

