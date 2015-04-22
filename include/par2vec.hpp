#ifndef PAR2VEC_HPP_
#define PAR2VEC_HPP_

#include "common.hpp"
#include "corpus.hpp"

typedef float real;

class Parameters 
{
public:
  long long d;    // dimensionality of word vectors
  long long V;    // the number of words
  long long M;    // the number of paragraphs
  int wn;         // window size
  real lr;        // initial learning rate
  int dm;         // use distributed memory model (PV-DM)
  int dbow;       // use distributed bag or words (PV-DBOW)
  int hs;         // use hierarchical softmax if 1
  real sample;    // sampling probability
  int negative;   // the number of negative samples
  int num_iters;  // the number of iterations over train data
};

class Par2Vec
{
public:
  Par2Vec(Corpus &corpus,
            long long wordvec_dim,
            int window_size,
            real learning_rate,
            int use_dm,
            int use_dbow,
            int use_hs,
            int negative,
            real sample,
            int num_iters,
            int num_threads,
            int verbose);

  ~Par2Vec();
  void start_train();
  void save(std::string filepath);

private:
  Corpus &m_Corpus;
  std::vector<vocab_ptr>& m_vocabulary;
  Parameters m_params;
  int* m_table;

  const int table_size = 1e+8;
  real m_sample;
  int m_num_iters;
  int m_num_threads;
  int m_verbose;
  std::chrono::time_point<std::chrono::system_clock> m_start_time;
  std::vector<std::string> split(const std::string &s, char delim, real sample_prob, unsigned long long &next_random);
  long long sentences_seen_actual;

  boost::shared_ptr<real []> D0;    // paragraph vecotrs
  boost::shared_ptr<real []> U0;    // word vectors
  boost::shared_ptr<real []> U1;    // hierarchical softmax output
  boost::shared_ptr<real []> U2;    // negative sampling output

  void run(int thread_id);
  void init(); 
  void createTable();
};

#endif

