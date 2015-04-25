#ifndef WORD2VEC_HPP_
#define WORD2VEC_HPP_

#include "common.hpp"
#include "corpus.hpp"

typedef float real;

class Parameters 
{
public:
  long long d;    // dimensionality of word vectors
  long long V;    // the number of words
  int wn;         // window size
  real lr;        // initial learning rate
  int sg;         // use skipgram if 1 otherwise use continuos bag of words (cbow)
  int hs;         // use hierarchical softmax if 1
  real sample;    // sampling probability
  int negative;   // the number of negative samples
  int num_iters;  // the number of iterations over train data

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & d;
    ar & V;
    ar & wn;
    ar & lr;
    ar & sg;
    ar & hs;
    ar & sample;
    ar & negative;
    ar & num_iters;
  }
};

class Word2Vec
{
public:
  Word2Vec(Corpus &corpus,
            long long wordvec_dim,
            int window_size,
            real learning_rate,
            int use_skipgram,
            int use_hs,
            int negative,
            real sample,
            int num_iters,
            int num_threads,
            int verbose);

  ~Word2Vec();
  void start_train();
  void export_vectors(std::string filepath);

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
  std::vector<std::string> split(const std::string &s, char delim);
  long long words_seen_actual;

  boost::shared_ptr<real> U0;    // word vectors
  boost::shared_ptr<real> U1;    // hierarchical softmax output
  boost::shared_ptr<real> U2;    // negative sampling output

  void run(int thread_id);
  void init(); 
  void createTable();

  // Allow serialization to access non-public data members.  
  friend class boost::serialization::access; 

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    real *U0_,*U1_,*U2_;
    ar & m_params;     
    if(Archive::is_loading::value)
    {
      U0_ = new real[m_params.d * m_params.V];    
      U1_ = new real[m_params.d * m_params.V];    
      U2_ = new real[m_params.d * m_params.V];    

      U0.reset(U0_);
      U1.reset(U1_);
      U2.reset(U2_);
    }
    else
    {
      U0_ = U0.get();
      U1_ = U1.get();
      U2_ = U2.get();
    }
    ar & boost::serialization::make_array<real>(U0_,m_params.d*m_params.V);
    ar & boost::serialization::make_array<real>(U1_,m_params.d*m_params.V);
    ar & boost::serialization::make_array<real>(U2_,m_params.d*m_params.V);
    ar & m_vocabulary;
  }
};

#endif

