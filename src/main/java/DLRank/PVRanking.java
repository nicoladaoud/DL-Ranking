package DLRank;


import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.store.RAMDirectory;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class PVRanking {

    public static void main(String[] args) throws Exception {
        String query = "cheese pizza in America"; // some dummy query
        rankWithParagraphVectors(query);
    }

    /**
     * builds and returns a paragraph vector model that will be used to retrieve paragraph vectors.
     * @param iterator, the iterator over the search index
     * @param layerSize, number of neurons in the paragraph vector neural net.
     * @param learningRate, the learning rate of the neural network.
     * @return the trained PV neural network.
     */
    public static ParagraphVectors buildParagraphVectorModel(FieldValuesLabelAwareIterator iterator, int layerSize, double learningRate) {
        ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
                .iterate(iterator)
                .learningRate(learningRate)
                .layerSize(layerSize)
                .tokenizerFactory(new DefaultTokenizerFactory())
                .build();

        paragraphVectors.fit();
        return paragraphVectors;
    }

    /**
     * Uses paragraph vectors to rank documents against a certain query
     * @throws Exception
     */
    public static void rankWithParagraphVectors(String queryString) throws Exception {
        Map<String, Double> cosineSim = new HashMap<String, Double>();
        Directory directory = new RAMDirectory(); // use a ram directory for the purpose of this project
        IndexWriterConfig config = new IndexWriterConfig(new WhitespaceAnalyzer());
        IndexWriter writer = new IndexWriter(directory, config);
        final String fieldName = "body";
        FieldType fieldType = new FieldType(TextField.TYPE_STORED);
        fieldType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
        fieldType.setTokenized(true);
        fieldType.setStored(true);

        Document doc1 = new Document();
        doc1.add(new Field(fieldName, "Cheese pizza is most popular in the United States.", fieldType));

        Document doc2 = new Document();
        doc2.add(new Field(fieldName, "Americans love cheese pizza.", fieldType));

        Document doc3 = new Document();
        doc3.add(new Field(fieldName, "Dominos makes the best cheese pizza in the U.S.", fieldType));

        Document doc4 = new Document();
        doc4.add(new Field(fieldName, "Americans order pizza on average twice a week!", fieldType));

        Document doc5 = new Document();
        doc5.add(new Field(fieldName, "Southern America does not have any fast food restaurants.", fieldType));

        writer.addDocument(doc1);
        writer.addDocument(doc2);
        writer.addDocument(doc3);
        writer.addDocument(doc4);
        writer.addDocument(doc5);
        writer.commit();

        IndexReader reader = DirectoryReader.open(writer);

        FieldValuesLabelAwareIterator iterator = new FieldValuesLabelAwareIterator(reader, fieldName);

        ParagraphVectors paragraphVectors = buildParagraphVectorModel(iterator, 50, 0.001);

        try {
            IndexSearcher searcher = new IndexSearcher(reader);

            INDArray queryParagraphVector = paragraphVectors.getLookupTable().vector(queryString);

            if (queryParagraphVector == null) {
                queryParagraphVector = paragraphVectors.inferVector(queryString);
            }

            QueryParser parser = new QueryParser(fieldName, new WhitespaceAnalyzer());
            Query query = parser.parse(queryString);
            TopDocs hits = searcher.search(query, 5);
            for (ScoreDoc scoreDoc : hits.scoreDocs) {
                Document doc = searcher.doc(scoreDoc.doc);
                String body = doc.get(fieldName);
                System.out.println(body + " : " + scoreDoc.score);
                String label = "doc_" + scoreDoc.doc;
                INDArray documentParagraphVector = paragraphVectors.getLookupTable().vector(label);
                if (documentParagraphVector == null) {
                    LabelledDocument document = new LabelledDocument();
                    document.setLabels(Collections.singletonList(label));
                    document.setContent(body);
                    documentParagraphVector = paragraphVectors.inferVector(document);
                }
                double cosineSimQueryDoc = Transforms.cosineSim(queryParagraphVector, documentParagraphVector);
                cosineSim.put(body, cosineSimQueryDoc);
            }

        } finally {
            //WordVectorSerializer.writeParagraphVectors(paragraphVectors, "target/pv.zip");
            writer.deleteAll();
            writer.commit();
            writer.close();
            reader.close();
        }
        directory.close();
    }
}
