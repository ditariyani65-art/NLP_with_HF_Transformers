<h1 align="center"> Machine Learning Notebooks </h1>
<p align="center"> Jupyter Notebook untuk mempelajari tentang regresi, klasifikasi dan clustering</p>
---
<h3> Name : Dita Riyani </h3>
<h3> ### My todo :  </h3>

#### 1. Example 1 - Sentiment Analysis

```
# TODO :
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("This class is amazing and very helpful for students.")
```

Result : 

```
[{'label': 'POSITIVE', 'score': 0.9998764991760254}]
```

Analysis on example 1 :

Model analisis sentimen secara akurat memberikan prediksi label POSITIVE dengan tingkat keyakinan (confidence score) yang sangat tinggi, yaitu mencapai 99.98%. Hal ini menunjukkan bahwa model DistilBERT sangat sensitif dan andal dalam mengenali ekspresi kepuasan atau impresi positif yang kuat dari kalimat subjek, khususnya dalam konteks umpan balik (feedback) pembelajaran di kelas.


#### 2. Example 2 - Topic Classification

```
# TODO :
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "Artificial Intelligence and machine learning are transforming how we analyze big data and build automated systems.",
    candidate_labels=["data science", "cybersecurity", "web development"],
)
```

Result : 

```
{'sequence': 'Artificial Intelligence and machine learning are transforming how we analyze big data and build automated systems.',
 'labels': ['data science', 'cybersecurity', 'web development'],
 'scores': [0.8563293814659119, 0.09478073567152023, 0.048889823257923126]}
```

Analysis on example 2 : 

Model Zero-Shot Classification menggunakan arsitektur bart-large-mnli berhasil mengklasifikasikan teks dengan sangat akurat. Kalimat input yang membahas tentang AI, machine learning, big data, dan sistem otomatisasi berhasil dikenali sebagai topik data science dengan skor kepercayaan tertinggi mencapai 85.63%.

Sementara itu, topik cybersecurity mendapatkan 9.47% dan web development hanya mendapatkan 4.88%, yang membuktikan bahwa model mampu memprioritaskan kata kunci kontekstual (konteks data dan analitik) secara tepat dibanding topik IT lainnya.


#### 3. Example 3 and 3.5 - Text Generator

```
# TODO 3 :
generator = pipeline("text-generation", model="distilgpt2") # or change to gpt-2
generator(
    "Developing a robust machine learning model requires not only high quality data but also",
    max_length=15, # you can change this
    num_return_sequences=2, # and this too
)
```

Result : 

```
[{'generated_text': 'Developing a robust machine learning model requires not only high quality data but also a real-time, multi-user experience.\n\n\n\n\n\n\nThe research team focused on the role of the neural network in developing new, adaptive learning models. They also used the technique of learning in learning models of a variety of techniques, including the neural network.\nThe neural network has previously been described as a "supercomputer science" that is based on the techniques used to predict the behavior of a human being.\n"Now, we have real-time models, that show the characteristics of the brain in terms of learning patterns and the types of behaviors that the brain uses," said lead author Dr. Martin Beulce, a professor in the Department of Mechanical Engineering and a part of the team.\nThe team\'s work comes as the company expands its business and has created a new generation of AI-driven machine learning models.\n"Our model is already very important to the technology, the algorithms and the knowledge that people can build from it," said Beulce. "It\'s important to integrate into the machine learning model."\nThe team expects to launch a second study in 2016.\n"We plan to develop a very exciting new model that will help us develop the next generation of AI-driven machine learning models," Beulce said'},
 {'generated_text': 'Developing a robust machine learning model requires not only high quality data but also a knowledge of how to implement it in a well-designed, efficient way, but also a willingness to be part of a team that actively iterates over and over a large amount of data.\n\n\n\n\n\nThe new model may be used in some scenarios, for instance, to study the accuracy of the time period of a sample. As a result, it may be useful to find out whether the time period of a sample is a measure of the size of the sample. For example, if you are looking for the number of days that you have to spend in a given day, you might want to add the same number of days as you did in the previous time frame. For example, if you have a group of people who spend a lot of time working on a particular project, you may want to use your own time frame instead of a single day.\nAdditionally, if you are using a given day for a particular project, you may want to add a few more days to the data. For example, if you have a data set of people who spend a lot of time working on a specific project, you may want to use your own time frame instead of a single day.\nThe new model also allows you to use data from the same source'}]
```

Analysis on example 3 : 

indicates a configuration anomaly, where the output text length far exceeds the max_length=15 parameter limit specified in the code. Qualitatively, while the DistilGPT-2 model is capable of reproducing the initial prompt with natural grammar, it is prone to "hallucinations" by fabricating facts and fictional characters (as in sequence 1). Furthermore, the model also experiences context drift, which causes sentence repetition and looping logic as the text lengthens (as in sequence 2). Overall, DistilGPT-2 is very efficient at constructing grammar, but requires further parameter tuning to maintain accuracy and prevent repetition in long texts.

```
# TODO 3.5 :
unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("The network administrator configures the <mask> to block unauthorized traffic.", top_k=4)
```

Result :
```python
[{'score': 0.5493684411048889,
  'token': 38956,
  'token_str': ' firewall',
  'sequence': 'The network administrator configures the firewall to block unauthorized traffic.'},
 {'score': 0.14371109008789062,
  'token': 30577,
  'token_str': ' router',
  'sequence': 'The network administrator configures the router to block unauthorized traffic.'},
 {'score': 0.035330746322870255,
  'token': 1546,
  'token_str': ' network',
  'sequence': 'The network administrator configures the network to block unauthorized traffic.'},
 {'score': 0.02985772117972374,
  'token': 37579,
  'token_str': ' routers',
  'sequence': 'The network administrator configures the routers to block unauthorized traffic.'}]
```
  
Analysis on example 3.5 :

The fill-mask pipeline accurately infers technical terms based on the IT context provided in the sentence. The top result "firewall" makes perfect sense for the task of blocking unauthorized traffic, supported by a solid confidence score (0.54). Other predictions like "router", "network", and "routers" are also contextually relevant to network administration, demonstrating the model's strong capability in understanding specialized vocabulary and technical context. 

#### 4. Example 4 - Name Entity Recognition (NER)

```
# TODO :
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
ner("Sundar Pichai, the CEO of Google, announced a new AI research facility in London.")
```

Result : 

```
[{'entity_group': 'PER',
  'score': np.float32(0.9961995),
  'word': 'Sundar Pichai',
  'start': 0,
  'end': 13},
 {'entity_group': 'ORG',
  'score': np.float32(0.9991849),
  'word': 'Google',
  'start': 26,
  'end': 32},
 {'entity_group': 'MISC',
  'score': np.float32(0.43944654),
  'word': 'AI',
  'start': 50,
  'end': 52},
 {'entity_group': 'LOC',
  'score': np.float32(0.9990094),
  'word': 'London',
  'start': 74,
  'end': 80}]
```

Analysis on example 4: 

This experiment demonstrates that the bert-large-cased-finetuned-conll03-english model is highly reliable in automatically understanding context and classifying entities within a sentence. However, from a software engineering perspective, the resulting error underscores the importance of paying attention to versioning dependency management to prevent default parameters from crashing the system.

#### 5. Example 5 - Question Answering

```
# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "What does machine learning allow computers to do?"
context = "Machine learning is a subset of artificial intelligence that allows computers to learn from data and improve their performance over time without being explicitly programmed."
qa_model(question = question, context = context)
```

Result : 

```
{'score': 0.35888803005218506,
 'start': 101,
 'end': 136,
 'answer': 'improve their performance over time'}
```

Analysis on example 5 : 

This experiment demonstrates that the BERT-derived model (DistilBERT) is highly efficient at understanding semantic relationships between questions and text context. This method is ideal for building internal search engine features or document parsing chatbots, where the system must instantly find specific information from a stack of paragraphs or long articles.

#### 6. Example 6: Text Summarization

```
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer(
    """
    Data Science has emerged as one of the most transformative fields in the modern digital era, driving decision-making across global industries. 
    At its core, data science combines statistical methodology, advanced computing algorithms, and specialized domain expertise to extract meaningful insights from vast repositories of unstructured and structured data. 
    The primary lifecycle begins with data acquisition and pipeline engineering, followed by rigorous data cleansing to ensure data integrity and remove systemic bias. 
    Once the dataset is ready, data scientists utilize sophisticated machine learning models to identify hidden patterns, forecast market trends, and automate repetitive organizational tasks. 
    Python serves as the dominant programming language in this ecosystem due to its powerful open-source libraries, including NumPy for numerical arrays, Pandas for structured data manipulation, and Scikit-Learn for training robust predictive algorithms.
    """
)
```

Result : 

```
[{'summary_text': ' Data Science has emerged as one of the most transformative fields in the modern digital era, driving decision-making across global industries . Python serves as the dominant programming language in this ecosystem due to its powerful open-source libraries, including NumPy for numerical arrays, Pandas for structured data manipulation, and Scikit-Learn for training robust predictive algorithms .'}]
```

Analysis on example 6 :   

shows that the DistilBART model successfully identifies the essence of a given custom paragraph by extracting the opening sentence about the definition of Data Science and the closing sentence about Python as the main idea. Although the model is smart in filtering important information and discarding supplementary details, the output shows the issue of text truncation at the end of the sentence ("Python se"). This indicates that using the raw summarization pipeline without setting additional parameters (such as max_length and min_length) can produce incomplete summaries, especially if the length of the input text does not match the standard ratio of the built-in model. instantly.


#### 7. Example 7 - Translation

```
# TODO :
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("masukan kebaikan, keluaran kebahagian: membangun jiwa sosial dan mempererat ukhuwah bersama anak-anak panti di yayasan al-kahfi medan")
```

Result : 

```
[{'translation_text': "Inscrivez le bien, la sortie du bonheur: construire une âme sociale et s'agrandir avec les enfants dans les maisons de retraite de la fondation al-kafi."}]
```

Analysis on example 7 : 

The translation model delivers an accurate and context-aware French translation of the Indonesian sentence. It handles informal, conversational input smoothly, making it suitable for multilingual communication tasks and cross-language understanding in casual or daily scenarios.

<h3> Analisis terhadap proyek ini </h3>

This project concluded that while pre-trained NLP pipelines offer ease and speed of implementation, there is a significant trade-off in the susceptibility of models to "illusions of intelligence" such as information hallucinations and misinterpretation of local context. The various technical challenges that emerged, ranging from errors due to outdated parameters to automatic text truncation, emphasized that AI implementation requires careful software engineering control and hyperparameter management. Therefore, these off-the-shelf models proved inadequate for complex, domain-specific systems; real-world implementation on educational platforms simulating industrial-level IT professions would require further fine-tuning and guardrails using highly focused datasets.
