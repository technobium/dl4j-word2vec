package com.technobium.word2vec;

import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Collection;

public class Word2VecDemo {

    private String inputFilePath = "input/content.txt";
    private String modelFilePath = "output/word2vec.bin";

    public static void main(String[] args) throws IOException {

        Word2VecDemo word2VecDemo = new Word2VecDemo();
        // Before starting the training, don't forget to add the text into input/content.txt
        word2VecDemo.train();

        Word2Vec word2VecModel = WordVectorSerializer.readWord2VecModel(new File(word2VecDemo.modelFilePath));

        Collection<String> list = word2VecModel.wordsNearest("boy" , 10);
        System.out.println(" boy: "+ list);

        list = word2VecModel.wordsNearest("girl" , 10);
        System.out.println("girl: " + list);

        Collection<String> stringList = word2VecModel.wordsNearest("day", 10);
        System.out.println(" day: " + stringList);
    }

    public  void train() throws IOException {
        SentenceIterator sentenceIterator = new FileSentenceIterator(new File(inputFilePath));
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(2)
                .layerSize(300)
                .windowSize(5)
                .seed(42)
                .epochs(3)
                .elementsLearningAlgorithm(new SkipGram<VocabWord>())
                .iterate(sentenceIterator)
                .tokenizerFactory(tokenizerFactory)
                .build();
        vec.fit();

        WordVectorSerializer.writeWordVectors(vec, "output/word2vec.bin");
    }
}
