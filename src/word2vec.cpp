#include "word2vec.hpp"
#include "utils.hpp"

Word2Vec::Word2Vec(Corpus &corpus,
            long long wordvec_dim,
            int window_size,
            real learning_rate,
            int use_skipgram,
            int use_hs,
            int negative,
            real sample,
            int num_iters,
            int num_threads,
            int verbose)
      : m_Corpus(corpus),
        m_vocabulary(corpus.getVocabulary()),
        m_sample(sample),
        m_num_iters(num_iters),
        m_num_threads(num_threads),
        m_verbose(verbose),
        sentences_seen_actual(0),
        U0(NULL),
        U1(NULL),
        U2(NULL)
{
  // Initialize parameters
  m_params.d = wordvec_dim;
  m_params.V = m_Corpus.get_vocabsize();
  m_params.wn = window_size;
  m_params.lr = learning_rate;
  m_params.sg = use_skipgram;
  m_params.hs = use_hs;
  m_params.sample = m_sample;
  m_params.negative = negative;
  m_params.num_iters = num_iters;
}

Word2Vec::~Word2Vec()
{
  delete[] m_table;
}

void Word2Vec::start_train()
{
  init();

  m_start_time = std::chrono::system_clock::now();

  boost::thread_group threads;
  for (int i = 0; i < m_num_threads; i++)
  {
    threads.create_thread(boost::bind(&Word2Vec::run, this, i));
  }
  threads.join_all();
}

void Word2Vec::init()
{
  LOG(INFO) << "The number of lines in the corpus: " << m_Corpus.get_num_lines();
  long long a, b;

  LOG(INFO) << "Initializing the model";

  boost::mt19937 rng; 
  boost::normal_distribution<> nd(0.0, 1.0);
  boost::variate_generator<boost::mt19937&, 
                           boost::normal_distribution<> > boost_randn(rng, nd);

  U0.reset(new real [m_params.d * m_params.V]);
  CHECK(U0) << "Memory allocation failed: " 
            << m_params.d << " x " 
            << m_params.V << " (" 
            << (m_params.d*m_params.V*sizeof(real)/(real)1000000) << " MB)";

  for(a = 0; a < m_params.V; a++)
  {
    for(b = 0; b < m_params.d; b++)
    {
      U0[b+a*m_params.d] = boost_randn()/sqrt(m_params.V);
    }
  }

  U1.reset(new real [m_params.d*m_params.V]);
  CHECK(U1) << "Memory allocation failed: " 
            << m_params.d << " x " 
            << m_params.V << " (" 
            << (m_params.d*m_params.V*sizeof(real)/(real)1000000) << " MB)";

  for(a = 0; a < m_params.V; a++)
  {
    for(b = 0; b < m_params.d; b++)
    {
      U1[b+a*m_params.d] = boost_randn()/sqrt(m_params.V);
    }
  }

  U2.reset(new real [m_params.d*m_params.V]);
  CHECK(U2) << "Memory allocation failed: " 
            << m_params.d << " x " 
            << m_params.V << " (" 
            << (m_params.d*m_params.V*sizeof(real)/(real)1000000) << " MB)";

  for(a = 0; a < m_params.V; a++)
  {
    for(b = 0; b < m_params.d; b++)
    {
      U2[b+a*m_params.d] = boost_randn()/sqrt(m_params.V);
    }
  }

  createTable();
}

inline real sigmoid(real x)
{
  return 1/(1+exp(-x));
}

void Word2Vec::run(int thread_id)
{
  std::ifstream file;
  real lr = m_params.lr;
  unsigned long long next_random = (long long) thread_id;
  const long long M = m_Corpus.get_num_lines();
  const long long max_sentences = (long long) ceil(M/(real)m_num_threads);
  long long sentences_seen,last_sentences_seen;

  real* hid = new real[m_params.d];
  real* grad = new real[m_params.d];

  for(int iter = 0; iter < m_params.num_iters; ++iter)
  {
    file.open(m_Corpus.get_corpus_filename());

    CHECK(file.is_open()) << "Failed to open the file: " << m_Corpus.get_corpus_filename();

    long long inst_idx=0;
    std::string line;
    while(1)
    {
      if((inst_idx == max_sentences*thread_id) || inst_idx >= M) break;
      std::getline(file, line);
      ++inst_idx;
    }
    LOG(INFO) << "Thread " << thread_id << " started after reading " << inst_idx << " lines at epoch " << iter + 1;
    sentences_seen = 0; last_sentences_seen = 0;
    if(max_sentences*thread_id < M)
    {
      while(1)
      {
        if(sentences_seen >= max_sentences) break;
        std::getline(file, line);

        std::vector<std::string> tokens = split(line.c_str(), ' ', m_sample, next_random);

        sentences_seen++;
        if(sentences_seen - last_sentences_seen > 1000)
        {
          sentences_seen_actual += sentences_seen - last_sentences_seen;
          last_sentences_seen = sentences_seen;
          if (m_verbose)
          {
            std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
            fprintf(stdout,"%cAlpha: %f Progress: %.2f%% Instances/thread/sec: %.2f", 13, lr, 
              sentences_seen_actual / (real)M * 100,
              sentences_seen_actual / (real) std::chrono::duration_cast<std::chrono::seconds>(now-m_start_time).count());
            fflush(stdout);
          }
          lr = m_params.lr * ( 1 - sentences_seen_actual / (real) (M * m_params.num_iters));
          if (lr < m_params.lr * 0.0001) lr = m_params.lr * 0.0001;
        }

        if(m_params.sg)
        {
          // skip-gram, hierarchical softmax and/or negative sampling
          for(int pos=0; pos < (int) tokens.size(); ++pos)
          {
            long long input_word_idx = m_Corpus.getIndexOf(tokens[pos]);
            long long output_word_idx = -1;

            for(int j=pos-(m_params.wn)/2; j < pos+(m_params.wn)/2+1; ++j)
            {
              if (j == pos) continue;
              if (j < 0 || j >= (int) tokens.size()) continue;
              output_word_idx = m_Corpus.getIndexOf(tokens[j]);
              memset(grad, 0, m_params.d*sizeof(real));
              if (m_params.hs)
              {
                for(int k = 0; k < m_vocabulary[output_word_idx]->get_codelen(); ++k)
                {
                  long long inner_node_idx = m_vocabulary[output_word_idx]->get_inner_node_idxAt(k);
                  real f = sigmoid(cblas_sdot(m_params.d, 
                                              &U0[input_word_idx*m_params.d], 1,
                                              &U1[inner_node_idx*m_params.d], 1)
                                  ); 

                  int code = m_vocabulary[output_word_idx]->get_codeAt(k);
                  real delta = (1 - code - f) * lr;
                  cblas_saxpy(m_params.d, delta, &U1[inner_node_idx*m_params.d], 1, grad, 1);
                  cblas_saxpy(m_params.d, delta, &U0[input_word_idx*m_params.d], 1, &U1[inner_node_idx*m_params.d], 1);
                }
              }
              if (m_params.negative)
              {
                int label;
                long long target_word_idx;
                for(int k = 0; k < m_params.negative + 1; ++k)
                {
                  if (k == 0)
                  {
                    target_word_idx = input_word_idx;
                    label = 1;
                  }
                  else
                  {
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    target_word_idx = m_table[(next_random >> 16) % table_size];
                    if (target_word_idx == 0) target_word_idx = next_random % (m_params.V - 1) + 1;
                    if (target_word_idx == input_word_idx) continue;
                    label = 0;
                  }
                  real f = sigmoid(cblas_sdot(m_params.d, 
                                              &U0[input_word_idx*m_params.d], 1,
                                              &U2[target_word_idx*m_params.d], 1)
                                  ); 

                  real delta = (label - f) * lr;
                  cblas_saxpy(m_params.d, delta, &U2[target_word_idx*m_params.d], 1, grad, 1);
                  cblas_saxpy(m_params.d, delta, &U0[input_word_idx*m_params.d], 1, &U2[target_word_idx*m_params.d], 1);
                }
              }
              cblas_saxpy(m_params.d, 1, grad, 1, &U0[input_word_idx*m_params.d], 1);
            }
          }
        }
        else
        {
          // continuous bag of words, hierarchical softmax and/or negative sampling
          for(int pos=0; pos < (int) tokens.size(); ++pos)
          {
            long long output_word_idx = m_Corpus.getIndexOf(tokens[pos]);
            long long input_word_idx;

            memset(hid, 0, m_params.d*sizeof(real));
            for(int j=pos-(m_params.wn)/2; j < pos+(m_params.wn)/2+1; ++j)
            {
              if (j == pos) continue;
              if (j < 0 || j >= (int) tokens.size()) continue;
              input_word_idx = m_Corpus.getIndexOf(tokens[j]);
              cblas_saxpy(m_params.d, 1, &U0[input_word_idx*m_params.d], 1, hid, 1);
            }
            memset(grad, 0, m_params.d*sizeof(real));
            if (m_params.hs)
            {
              for(int k = 0; k < m_vocabulary[output_word_idx]->get_codelen(); ++k)
              {
                long long inner_node_idx = m_vocabulary[output_word_idx]->get_inner_node_idxAt(k);
                real f = sigmoid(cblas_sdot(m_params.d, 
                                            hid, 1,
                                            &U1[inner_node_idx*m_params.d], 1)
                                ); 

                int code = m_vocabulary[output_word_idx]->get_codeAt(k);
                real delta = (1 - code - f) * lr;
                cblas_saxpy(m_params.d, delta, &U1[inner_node_idx*m_params.d], 1, grad, 1);
                cblas_saxpy(m_params.d, delta, hid, 1, &U1[inner_node_idx*m_params.d], 1);
              }
            }
            if (m_params.negative)
            {
              int label;
              long long target_word_idx;
              for(int k = 0; k < m_params.negative + 1; ++k)
              {
                if (k == 0)
                {
                  target_word_idx = output_word_idx;
                  label = 1;
                }
                else
                {
                  next_random = next_random * (unsigned long long)25214903917 + 11;
                  target_word_idx = m_table[(next_random >> 16) % table_size];
                  if (target_word_idx == 0) target_word_idx = next_random % (m_params.V - 1) + 1;
                  if (target_word_idx == output_word_idx) continue;
                  label = 0;
                }
                real f = sigmoid(cblas_sdot(m_params.d, 
                                            hid, 1,
                                            &U2[target_word_idx*m_params.d], 1)
                                ); 

                real delta = (label - f) * lr;
                cblas_saxpy(m_params.d, delta, &U2[target_word_idx*m_params.d], 1, grad, 1);
                cblas_saxpy(m_params.d, delta, hid, 1, &U2[target_word_idx*m_params.d], 1);
              }
            }
            for(int j=pos-(m_params.wn)/2; j < pos+(m_params.wn)/2+1; ++j)
            {
              if (j == pos) continue;
              if (j < 0 || j >= (int) tokens.size()) continue;
              input_word_idx = m_Corpus.getIndexOf(tokens[j]);
              cblas_saxpy(m_params.d, 1, grad, 1, &U0[input_word_idx*m_params.d], 1);
            }
          }
        }
      }
    }
    file.close();
  }

  delete[] grad;
  delete[] hid;
  LOG(INFO) << "Thread " << thread_id << " has finished.";
}

void Word2Vec::createTable()
{
  long long train_words_pow = 0;
  real d1, power = 0.75;

  m_table = new int[table_size];

  for (long long a = 0; a < m_params.V; a++) train_words_pow += pow(m_vocabulary[a]->m_freq, power);
  int i = 0;
  d1 = pow(m_vocabulary[i]->m_freq, power) / (real)train_words_pow;

  for (long long a = 0; a < table_size; a++)
  {
    m_table[a] = i;
    if (a / (real)table_size > d1)
    {
      i++;
      d1 += pow(m_vocabulary[i]->m_freq, power) / (real)train_words_pow;
    }
    if (i >= m_params.V) i = m_params.V - 1;
  }
}


std::vector<std::string> Word2Vec::split(const std::string &s, char delim, real sample_prob, unsigned long long &next_random)
{
  std::vector<std::string> elems;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim))
  {
    if (m_sample > 0)
    {
      // word index
      long long word_idx = m_Corpus.getIndexOf(item);
      real ran = (sqrt(m_vocabulary[word_idx]->m_freq / (m_sample * m_Corpus.get_train_words())) + 1) 
                  * (m_sample * m_Corpus.get_train_words()) / m_vocabulary[word_idx]->m_freq;
      next_random = next_random * (unsigned long long)25214903917 + 11;
      if (ran < (next_random & 0xFFFF) / (real)65536) continue;
    }
    elems.push_back(item);
  }
  return elems;
}

void Word2Vec::save(std::string filepath)
{
  // TODO store a complete model
  // At this stage, only word vectors are stored in a binary format.
  std::ofstream file(filepath, std::ios::out | std::ios::binary);

  CHECK(file.is_open()) << "Failed to open the file: " << filepath;

  file.write((char *) &(m_params.d), sizeof(long long));
  file.write((char *) &(m_params.V), sizeof(long long));
  file.write((char *) U0.get(), m_params.d*m_params.V*sizeof(real));
  file.close();

  LOG(INFO) << "Model parameters are stored under the following path: " << filepath;
}
