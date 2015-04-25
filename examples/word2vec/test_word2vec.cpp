#include "common.hpp"
#include "corpus.hpp"
#include "word2vec.hpp"

int main(int argc, char* argv[])
{
  google::InitGoogleLogging(argv[0]);

  std::string corpus_filepath("/data/Corpus/wikipedia_dump/de_wiki_sen_tokenized_lower_per_line.txt");
  int wordvec_dim = 100;
  int window_size = 11;
  float learning_rate = 0.025;
  int skipgram = 1;
  int hs = 1;
  int negative = 10;
  float sample = 1e-5;

  int num_iters = 10;
  int num_threads = 4;

  int verbose = 1;

  Corpus &corpus = Corpus::getCorpus(corpus_filepath);

  try{
    /*
    corpus.build_vocab();
		corpus.save_vocab(std::string("cbow_vocab.txt"));
    */
		corpus.load_vocab(std::string("cbow_vocab.txt"));
    corpus.create_huffman_tree();
  } catch (std::exception& e) {
    LOG(FATAL) << e.what();
  }

  Word2Vec model(corpus, wordvec_dim, window_size, learning_rate, skipgram, hs, negative, sample, num_iters, num_threads, verbose);
  model.start_train();

	LOG(INFO) << "Training done!";

  {
    const char* name = "word2vec_sg_trained_model.bin";
    std::ofstream out_stream(name);
    boost::archive::binary_oarchive oar(out_stream);
    oar << model;
    out_stream.close();
  }

	model.export_vectors(std::string("cbow_word_vectors_save.bin"));

  return 0;
}
