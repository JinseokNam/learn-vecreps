#include "common.hpp"
#include "corpus.hpp"
#include "par2vec.hpp"

int main(int argc, char* argv[])
{
  google::InitGoogleLogging(argv[0]);

  std::cout << "Testing Paragraph vectors" << std::endl;
  std::string corpus_filepath("/home/nam/codes/gensim_pipeline/new_wikipedia_all_paragraphs.txt");
  int vec_dim = 100;
  int window_size = 11;
  float learning_rate = 0.025;
  int dm = 0;
  int dbow = 1;
  int hs = 1;
  int negative = 10;
  float sample = 1e-5;

  int num_iters = 10;
  int num_threads = 24;

  int verbose = 1;

  Corpus &corpus = Corpus::getCorpus(corpus_filepath);

  try{
    /*
    corpus.build_vocab();
		corpus.save_vocab(std::string("dbow_vocab.txt"));
    */
		corpus.load_vocab(std::string("dbow_vocab.txt"));
    corpus.create_huffman_tree();
  } catch (std::exception& e) {
    LOG(FATAL) << e.what();
  }

  Par2Vec model(corpus, vec_dim, window_size, learning_rate, dm, dbow, hs, negative, sample, num_iters, num_threads, verbose);

  model.start_train();

	LOG(INFO) << "Training done!";

  {
    const char* name = "par2vec_dbow_trained_model.bin";
    std::ofstream out_stream(name);
    //boost::archive::text_oarchive oar(out_stream);
    boost::archive::binary_oarchive oar(out_stream);
    oar << model;
    out_stream.close();
  }

	model.export_vectors(std::string("dbow_word_paragraph_vectors.bin"));

  return 0;
}
