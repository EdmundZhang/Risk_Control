/*
需求：
我希望使用n-gram 模型，计算一个字符串的生成概率，用来寻找随机生成的字符串。
输入是邮箱数据，例如123@gmail.com,去除@以及@后面的域名，剩下的作为输入单词，使用ngram模型计算生成概率。使用hive sql实现。

功能实现：
1.首先创建了一个名为email_table的表格，用于存储邮箱数据。
2.使用regexp_replace函数将邮箱域名替换为空格，并使用split函数对文本进行分词。分词结果被存储在名为word_table的表格中。
3.对word_table中的单词数据进行n-gram处理，并统计每个n-gram出现的次数。
4.我们计算每个n-gram的概率，并将结果存储在名为prob_table的表格中。可以通过查询prob_table来计算给定字符串的生成概率。
*/

-- 创建一个名为email_table的表格，用于存储邮箱数据
CREATE TABLE email_table (
  email STRING,
  text STRING
);

-- 创建一个名为word_table的表格，用于存储单词数据
CREATE TABLE word_table (
  word STRING
);

-- 使用regexp_replace函数将邮箱域名替换为空格，并使用split函数对文本进行分词
INSERT INTO TABLE word_table
SELECT explode(split(lower(regexp_replace(text, '@[a-zA-Z0-9.-]+', '')))) AS word
FROM email_table;

-- 创建一个名为ngram_table的表格，用于存储n-gram模型中每个n-gram的出现次数
CREATE TABLE ngram_table (
  ngram STRING,
  count INT
);

-- 对单词数据进行n-gram处理，并统计每个n-gram出现的次数
INSERT INTO TABLE ngram_table
SELECT CONCAT_WS(' ', t1.word, t2.word, t3.word) AS ngram, COUNT(*) AS count
FROM (
  SELECT word, pos
  FROM word_table
) t1
JOIN (
  SELECT word, pos
  FROM word_table
) t2 ON t1.pos + 1 = t2.pos
JOIN (
  SELECT word, pos
  FROM word_table
) t3 ON t2.pos + 1 = t3.pos
GROUP BY CONCAT_WS(' ', t1.word, t2.word, t3.word);

-- 计算每个n-gram的概率，并将结果存储在名为prob_table的表格中
CREATE TABLE prob_table AS
SELECT ngram, count / (SELECT SUM(count) FROM ngram_table) AS prob
FROM ngram_table;

