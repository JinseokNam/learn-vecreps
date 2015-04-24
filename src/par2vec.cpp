#include "par2vec.hpp"
#include "utils.hpp"

Par2Vec::Par2Vec(Corpus &corpus,
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
            int verbose)
      : m_Corpus(corpus),
        m_vocabulary(corpus.getVocabulary()),
        m_sample(sample),
        m_num_iters(num_iters),
        m_num_threads(num_threads),
        m_verbose(verbose),
        words_seen_actual(0),
        D0(NULL),
        U0(NULL),
        U1(NULL),
        U2(NULL)
{
  // Initialize parameters
  m_params.d = wordvec_dim;
  m_params.V = m_Corpus.get_vocabsize();
  m_params.M = m_Corpus.get_num_lines();
  m_params.wn = window_size;
  m_params.lr = learning_rate;
  m_params.dm = use_dm;
  m_params.dbow = use_dbow;
  m_params.hs = use_hs;
  m_params.sample = m_sample;
  m_params.negative = negative;
  m_params.num_iters = num_iters;

  CHECK(use_dm ^ use_dbow) << "Allows to select either DM or DBOW";

  init();
}

Par2Vec::~Par2Vec()
{
  delete[] m_table;
}

void Par2Vec::start_train()
{
  m_start_time = std::chrono::system_clock::now();

  boost::thread_group threads;
  for (int i = 0; i < m_num_threads; i++)
  {
    threads.create_thread(boost::bind(&Par2Vec::run, this, i));
  }
  threads.join_all();
}

void Par2Vec::init()
{
  LOG(INFO) << "The number of lines in the corpus: " << m_Corpus.get_num_lines();
  long long a, b;

  LOG(INFO) << "Initializing the model";

  boost::mt19937 rng; 
  boost::normal_distribution<> nd(0.0, 1.0);
  boost::variate_generator<boost::mt19937&, 
                           boost::normal_distribution<> > boost_randn(rng, nd);

  real *D0_ = new real[m_params.d * m_params.M];
  CHECK(D0_) << "Memory allocation failed: " 
            << m_params.d << " x " 
            << m_params.M << " (" 
            << (m_params.d*m_params.M*sizeof(real)/(real)1000000) << " MB)";

  for(a = 0; a < m_params.M; a++)
  {
    for(b = 0; b < m_params.d; b++)
    {
      D0_[b+a*m_params.d] = boost_randn()/sqrt(m_params.M);
    }
  }

  real *U0_ = new real [m_params.d * m_params.V];
  CHECK(U0_) << "Memory allocation failed: " 
            << m_params.d << " x " 
            << m_params.V << " (" 
            << (m_params.d*m_params.V*sizeof(real)/(real)1000000) << " MB)";

  for(a = 0; a < m_params.V; a++)
  {
    for(b = 0; b < m_params.d; b++)
    {
      U0_[b+a*m_params.d] = boost_randn()/sqrt(m_params.V);
    }
  }

  real *U1_ = new real [m_params.d*m_params.V];
  CHECK(U1_) << "Memory allocation failed: " 
            << m_params.d << " x " 
            << m_params.V << " (" 
            << (m_params.d*m_params.V*sizeof(real)/(real)1000000) << " MB)";

  for(a = 0; a < m_params.V; a++)
  {
    for(b = 0; b < m_params.d; b++)
    {
      U1_[b+a*m_params.d] = boost_randn()/sqrt(m_params.V);
    }
  }

  real *U2_ = new real [m_params.d*m_params.V];
  CHECK(U2_) << "Memory allocation failed: " 
            << m_params.d << " x " 
            << m_params.V << " (" 
            << (m_params.d*m_params.V*sizeof(real)/(real)1000000) << " MB)";

  for(a = 0; a < m_params.V; a++)
  {
    for(b = 0; b < m_params.d; b++)
    {
      U2_[b+a*m_params.d] = boost_randn()/sqrt(m_params.V);
    }
  }

  D0.reset(D0_);
  U0.reset(U0_);
  U1.reset(U1_);
  U2.reset(U2_);

  createTable();
}

inline real sigmoid(real x)
{
  return 1/(1+exp(-x));
}

void Par2Vec::run(int thread_id)
{
  std::ifstream file;
  real lr = m_params.lr;
  unsigned long long next_random = (long long) thread_id;
  const long long M = m_Corpus.get_num_lines();
  const long long total_num_words = m_Corpus.get_train_words();
  const long long max_words = (long long) ceil(total_num_words/(real)m_num_threads);
  long long words_seen,last_words_seen,sentences_seen;
  long long base_sentence_idx, num_words_read;

  real* hid = new real[m_params.d];
  real* grad = new real[m_params.d];
  real *D0_, *U0_, *U1_, *U2_;

  D0_ = D0.get();
  U0_ = U0.get();
  U1_ = U1.get();
  U2_ = U2.get();

  file.open(m_Corpus.get_corpus_filename());

  CHECK(file.is_open()) << "Failed to open the file: " << m_Corpus.get_corpus_filename();

  base_sentence_idx=0;
  num_words_read = 0;
  std::string line;
  while(1)
  {
    if((num_words_read >= max_words*thread_id) || base_sentence_idx >= M) break;
    std::getline(file, line);
    num_words_read += (std::count(line.begin(), line.end(), ' ') + 1);
    ++base_sentence_idx;
  }
  file.close();

  for(int iter = 0; iter < m_params.num_iters; ++iter)
  {
    file.open(m_Corpus.get_corpus_filename());
    num_words_read = 0;
    long long sentence_count = 0;
    while(1)
    {
      if(sentence_count == base_sentence_idx) break;
      if(!std::getline(file, line)) break;
      sentence_count++;
    }

    //LOG(INFO) << "Thread " << thread_id << " started after reading " << base_sentence_idx << " lines at epoch " << iter + 1;

    words_seen = 0; last_words_seen = 0; sentences_seen = 0;
    while(1)
    {
      if(words_seen >= max_words) break;
      if(!std::getline(file, line)) break;

      std::vector<std::string> tokens = split(line.c_str(), ' ', m_sample, next_random);

      words_seen += (std::count(line.begin(), line.end(), ' ') + 1);
      sentences_seen++;
      if(words_seen - last_words_seen > 100000)
      {
        words_seen_actual += words_seen - last_words_seen;
        last_words_seen = words_seen;
        if (m_verbose)
        {
          std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
          fprintf(stdout,"%cLearningRate: %f Progress: %.2f%% Processing words/sec: %.2f k", 13, lr, 
            words_seen_actual / (real)total_num_words * 100,
            words_seen_actual / (real) std::chrono::duration_cast<std::chrono::seconds>(now-m_start_time).count() / 1000.);
          fflush(stdout);
        }
        lr = m_params.lr * ( 1 - words_seen_actual / (real) (total_num_words * m_params.num_iters));
        if (lr < m_params.lr * 0.0001) lr = m_params.lr * 0.0001;
      }

      long long paragraph_idx = base_sentence_idx + sentences_seen - 1;
      CHECK(paragraph_idx >= 0 && paragraph_idx < M) << "Illegal index of paragraph: " << paragraph_idx << " out of " << M << " Thread id: " << thread_id;

      if(m_params.dm)
      {
        // distributed memory model, hierarchical softmax and/or negative sampling
        for(int pos=0; pos < (int) tokens.size(); ++pos)
        {
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
                                            D0_+paragraph_idx*m_params.d, 1,
                                            U1_+inner_node_idx*m_params.d, 1)
                                ); 
                CHECK(!isnan(f) && !isinf(f));

                int code = m_vocabulary[output_word_idx]->get_codeAt(k);
                real delta = (1 - code - f) * lr;
                cblas_saxpy(m_params.d, delta, U1_+inner_node_idx*m_params.d, 1, grad, 1);
                cblas_saxpy(m_params.d, delta, D0_+paragraph_idx*m_params.d, 1, U1_+inner_node_idx*m_params.d, 1);
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
                  target_word_idx = paragraph_idx;
                  label = 1;
                }
                else
                {
                  next_random = next_random * (unsigned long long)25214903917 + 11;
                  target_word_idx = m_table[(next_random >> 16) % table_size];
                  if (target_word_idx == 0) target_word_idx = next_random % (m_params.V - 1) + 1;
                  label = 0;
                }
                real f = sigmoid(cblas_sdot(m_params.d, 
                                            D0_+paragraph_idx*m_params.d, 1,
                                            U2_+target_word_idx*m_params.d, 1)
                                ); 
                CHECK(!isnan(f) && !isinf(f));

                real delta = (label - f) * lr;
                cblas_saxpy(m_params.d, delta, U2_+target_word_idx*m_params.d, 1, grad, 1);
                cblas_saxpy(m_params.d, delta, D0_+paragraph_idx*m_params.d, 1, U2_+target_word_idx*m_params.d, 1);
              }
            }
            cblas_saxpy(m_params.d, 1, grad, 1, D0_+paragraph_idx*m_params.d, 1);
          }
        }
      }
      else if(m_params.dbow)
      {
        // distributed bag of words, hierarchical softmax and/or negative sampling
        for(int pos=0; pos < (int) tokens.size(); ++pos)
        {
          cblas_scopy(m_params.d, D0_+paragraph_idx*m_params.d, 1, hid, 1);
          long long output_word_idx = m_Corpus.getIndexOf(tokens[pos]);
          long long input_word_idx;
          int num_components = 1;

          for(int j=pos-(m_params.wn)/2; j < pos+(m_params.wn)/2+1; ++j)
          {
            if (j == pos) continue;
            if (j < 0 || j >= (int) tokens.size()) continue;
            input_word_idx = m_Corpus.getIndexOf(tokens[j]);
            cblas_saxpy(m_params.d, 1, U0_+input_word_idx*m_params.d, 1, hid, 1);
            num_components++;
          }
          cblas_sscal(m_params.d, 1/(real)num_components, hid, 1);
          memset(grad, 0, m_params.d*sizeof(real));
          if (m_params.hs)
          {
            for(int k = 0; k < m_vocabulary[output_word_idx]->get_codelen(); ++k)
            {
              long long inner_node_idx = m_vocabulary[output_word_idx]->get_inner_node_idxAt(k);
              real f = sigmoid(cblas_sdot(m_params.d, 
                                          hid, 1,
                                          U1_+inner_node_idx*m_params.d, 1)
                              ); 
              CHECK(!isnan(f) && !isinf(f));

              int code = m_vocabulary[output_word_idx]->get_codeAt(k);
              real delta = (1 - code - f) * lr;
              cblas_saxpy(m_params.d, delta, U1_+inner_node_idx*m_params.d, 1, grad, 1);
              cblas_saxpy(m_params.d, delta, hid, 1, U1_+inner_node_idx*m_params.d, 1);
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
                                          U2_+target_word_idx*m_params.d, 1)
                              ); 
              CHECK(!isnan(f) && !isinf(f));

              real delta = (label - f) * lr;
              cblas_saxpy(m_params.d, delta, U2_+target_word_idx*m_params.d, 1, grad, 1);
              cblas_saxpy(m_params.d, delta, hid, 1, U2_+target_word_idx*m_params.d, 1);
            }
          }
          for(int j=pos-(m_params.wn)/2; j < pos+(m_params.wn)/2+1; ++j)
          {
            if (j == pos) continue;
            if (j < 0 || j >= (int) tokens.size()) continue;
            input_word_idx = m_Corpus.getIndexOf(tokens[j]);
            cblas_saxpy(m_params.d, 1/(real)num_components, grad, 1, U0_+input_word_idx*m_params.d, 1);
          }
          cblas_saxpy(m_params.d, 1/(real)num_components, grad, 1, D0_+paragraph_idx*m_params.d, 1);
        }
      }
    }
    // LOG(INFO) << "Last paragraph index: " << (base_sentence_idx + sentences_seen - 1) << " in Thread id: " << thread_id;
    file.close();
  }

  delete[] grad;
  delete[] hid;
  LOG(INFO) << "Thread " << thread_id << " has finished.";
}

void Par2Vec::createTable()
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


std::vector<std::string> Par2Vec::split(const std::string &s, char delim, real sample_prob, unsigned long long &next_random)
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

std::vector<std::string> Par2Vec::split(const std::string &s, char delim)
{
  std::vector<std::string> elems;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

void Par2Vec::export_vectors(std::string filepath)
{
  // store word vectors and paragraph vectors to the designated path
  std::ofstream file(filepath, std::ios::out | std::ios::binary);

  CHECK(file.is_open()) << "Failed to open the file: " << filepath;

  file.write((char *) &(m_params.d), sizeof(long long));
  file.write((char *) &(m_params.V), sizeof(long long));
  file.write((char *) U0.get(), m_params.d*m_params.V*sizeof(real));

  file.write((char *) &(m_params.d), sizeof(long long));
  file.write((char *) &(m_params.M), sizeof(long long));
  file.write((char *) D0.get(), m_params.d*m_params.M*sizeof(real));

  file.close();

  LOG(INFO) << "Model parameters are stored under the following path: " << filepath;
}
