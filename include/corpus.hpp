#ifndef CORPUS_HPP_
#define CORPUS_HPP_

#include "common.hpp"
#include "vocab.hpp"

typedef boost::shared_ptr<Vocab> vocab_ptr;

class Corpus
{
public:
  static Corpus& getCorpus(std::string corpus_filename)
  {
    static Corpus instance(corpus_filename);
    return instance;
  }
  void build_vocab();
  void save_vocab(std::string filepath);
  void create_huffman_tree(); 
  std::string get_corpus_filename() const;
  long long get_vocabsize() const;
  long long get_num_lines() const;
  long long getIndexOf(std::string &word);
  long long getUNKIndex() const;

  std::vector<vocab_ptr>& getVocabulary();

  int getCodeLen(long long word_idx) const;
  long long getInnerIndexOfCodeAt(long long word_idx, int k);
  int get_codeAt(int i);

  long long get_train_words() const;


private:
  Corpus();
  Corpus(std::string corpus_filename);
  Corpus(Corpus const&);
  void sortVocab();

  ~Corpus();

  std::string m_corpus_filename;
  std::ifstream m_corpus_file;
  int m_min_count;
  std::vector<vocab_ptr> m_vocabulary;
  long long m_num_lines;
  std::map<std::string,long long> m_word2idx_map;
  long long m_train_words;
};

#endif
