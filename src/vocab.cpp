#include "vocab.hpp"

Vocab::Vocab()
      : m_word(""),
        m_freq(-1),
        m_codelen(-1),
        m_codes(0),
        m_inner_node_idx(0)
{
}

Vocab::Vocab(std::string word, int freq) 
      : m_word(word),
        m_freq(freq),
        m_codelen(-1),
        m_codes(0),
        m_inner_node_idx(0)
{
}

Vocab::~Vocab()
{
}

void Vocab::set_codeword(char *codeword)
{
  m_codes.clear();
  for(unsigned int i = 0; i < strlen(codeword); i++)
  {
    m_codes.push_back(codeword[i]);  
  }
}

const char* Vocab::get_codeword()
{
  std::string s(m_codes.begin(),m_codes.end());
  s.push_back('\0');
  return s.c_str();
}

int Vocab::get_codeAt(int i)
{
  return m_codes[i] - '0';
}

int Vocab::get_codelen() const
{
  const int ret = m_codes.size();
  return ret;
}

void Vocab::set_inner_node_idx(std::vector<long long> inner_node_idx)
{
  CHECK_EQ(inner_node_idx.size(), m_codes.size());

  m_inner_node_idx.clear();
  for(auto& node_idx : inner_node_idx)
  {
    m_inner_node_idx.push_back(node_idx);
  }
}

const char* Vocab::get_inner_node_idx()
{
  std::stringstream ss;
  std::copy(m_inner_node_idx.begin(), m_inner_node_idx.end(), std::ostream_iterator<long long>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);
  s.push_back('\0');
  return s.c_str();
}

long long Vocab::get_inner_node_idxAt(int i)
{
  return m_inner_node_idx[i];
}
