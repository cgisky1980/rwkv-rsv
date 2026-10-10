//! RWKV 分词器（从 web-rwkv 移植，去掉 wasm/ahash 依赖，改用 std 集合）。
//!
//! 词表格式与 web-rwkv 一致：一个 `{token_id: 字符串或字节数组}` 的 JSON 映射，
//! 例如官方的 `rwkv_vocab_v20230424.json`。编码采用最长匹配贪心。

use std::collections::{BTreeMap, HashMap};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("failed to parse vocabulary: {0}")]
    FailedToParseVocabulary(serde_json::Error),
    #[error("no matching token found")]
    NoMatchingTokenFound,
    #[error("out of range token: {0}")]
    OutOfRangeToken(u32),
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    first_bytes_to_lengths: Vec<Box<[u16]>>,
    bytes_to_token_index: HashMap<Vec<u8>, u32>,
    token_index_to_bytes: Vec<Vec<u8>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
enum StrOrBytes {
    Str(String),
    Bytes(Vec<u8>),
}

impl Tokenizer {
    pub fn new(vocab: &str) -> Result<Self, TokenizerError> {
        let map: BTreeMap<u32, StrOrBytes> =
            serde_json::from_str(vocab).map_err(TokenizerError::FailedToParseVocabulary)?;

        let list: Vec<(Vec<u8>, u32)> = map
            .into_iter()
            .map(|(token, pattern)| {
                let pattern: Vec<u8> = match pattern {
                    StrOrBytes::Str(string) => string.into_bytes(),
                    StrOrBytes::Bytes(bytes) => bytes,
                };
                (pattern, token)
            })
            .collect();

        let mut first_bytes_to_lengths = Vec::new();
        first_bytes_to_lengths.resize(u16::MAX as usize, {
            let mut set = std::collections::BTreeSet::new();
            set.insert(1);
            set
        });

        let mut token_index_to_bytes = Vec::new();
        let max_token_index = list.iter().map(|(_, index)| *index).max().unwrap_or(0) as usize;
        token_index_to_bytes.resize_with(max_token_index + 1, Vec::new);

        let mut bytes_to_token_index = HashMap::new();
        for (token_bytes, token_index) in list {
            if token_bytes.len() >= 2 {
                let key = u16::from_ne_bytes([token_bytes[0], token_bytes[1]]) as usize;
                first_bytes_to_lengths[key].insert(token_bytes.len() as u16);
            }

            bytes_to_token_index.insert(token_bytes.clone(), token_index);
            token_index_to_bytes[token_index as usize] = token_bytes;
        }

        let first_bytes_to_lengths: Vec<Box<[_]>> = first_bytes_to_lengths
            .into_iter()
            .map(|inner| {
                let mut inner: Vec<_> = inner.into_iter().collect();
                inner.sort_unstable_by_key(|l| !*l);
                inner.into_boxed_slice()
            })
            .collect();

        Ok(Tokenizer {
            first_bytes_to_lengths,
            bytes_to_token_index,
            token_index_to_bytes,
        })
    }

    pub fn encode(&self, input: &[u8]) -> Result<Vec<u32>, TokenizerError> {
        let mut output = Vec::new();
        self.encode_into(input, &mut output)?;
        Ok(output)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<Vec<u8>, TokenizerError> {
        let mut output = Vec::with_capacity(tokens.len());
        self.decode_into(tokens, &mut output)?;
        Ok(output)
    }

    pub fn encode_into(
        &self,
        mut input: &[u8],
        output: &mut Vec<u32>,
    ) -> Result<(), TokenizerError> {
        'next_token: while !input.is_empty() {
            let lengths = if input.len() >= 2 {
                let key = u16::from_ne_bytes([input[0], input[1]]) as usize;
                &self.first_bytes_to_lengths[key][..]
            } else {
                &[1][..]
            };

            for &length in lengths {
                let length = length as usize;
                if length > input.len() {
                    continue;
                }

                if let Some(&token_index) = self.bytes_to_token_index.get(&input[..length]) {
                    output.push(token_index);
                    input = &input[length..];
                    continue 'next_token;
                }
            }

            return Err(TokenizerError::NoMatchingTokenFound);
        }

        Ok(())
    }

    pub fn decode_into(&self, tokens: &[u32], output: &mut Vec<u8>) -> Result<(), TokenizerError> {
        for &token in tokens {
            let bytes = self
                .token_index_to_bytes
                .get(token as usize)
                .ok_or(TokenizerError::OutOfRangeToken(token))?;

            output.extend_from_slice(bytes);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 内联小词表，覆盖最长匹配、CJK、前导空格、字节数组词条与 id 空洞。
    /// 两种 `StrOrBytes` 形态都在里面：字符串（官方词表的样子）与字节数组
    /// （可表达非法 UTF-8 的裸字节）。
    const VOCAB: &str = r#"{
        "0": "a",
        "1": "b",
        "2": "ab",
        "3": "abc",
        "4": "北京",
        "5": "上",
        "6": "海",
        "7": "上海",
        "8": " Paris",
        "9": "Pari",
        "10": [226, 130, 136],
        "11": "猫",
        "12": "猫ab",
        "13": [128],
        "20": "zz"
    }"#;

    fn tokenizer() -> Tokenizer {
        Tokenizer::new(VOCAB).expect("内联词表必须能解析")
    }

    #[test]
    fn rejects_garbage_vocab() {
        let err = Tokenizer::new("这不是 JSON").unwrap_err();
        assert!(
            matches!(err, TokenizerError::FailedToParseVocabulary(_)),
            "坏词表必须是 FailedToParseVocabulary，实际：{err:?}"
        );
    }

    /// 最长匹配是本分词器的核心口径：`abc` 必须合成一个 token，而不是 `ab`+`c`。
    #[test]
    fn longest_match_wins() {
        let t = tokenizer();
        assert_eq!(t.encode(b"abc").unwrap(), vec![3]);
        assert_eq!(t.encode(b"ab").unwrap(), vec![2]);
        assert_eq!(t.encode(b"a").unwrap(), vec![0]);
        assert_eq!(t.encode(b"zz").unwrap(), vec![20]);
    }

    /// 贪心但不回溯：长词不命中就退到次长，剩余部分继续从当前位置匹配。
    #[test]
    fn falls_back_to_shorter_then_restarts() {
        let t = tokenizer();
        // "aba"：长度 3/2 的候选里只有 "ab" 命中，剩下的 "a" 单独成 token。
        assert_eq!(t.encode(b"aba").unwrap(), vec![2, 0]);
        assert_eq!(t.encode(b"abb").unwrap(), vec![2, 1]);
    }

    /// 末位只剩 1 字节时只尝试长度 1（`length > input.len()` 的跳过分支）。
    #[test]
    fn skips_lengths_exceeding_remaining_input() {
        let t = tokenizer();
        // "上" 的首两字节与 "上海" 同键，键表里是 [6, 3, 1]；6 必须先被跳过。
        assert_eq!(t.encode("上".as_bytes()).unwrap(), vec![5]);
        assert_eq!(t.encode("猫a".as_bytes()).unwrap(), vec![11, 0]);
    }

    /// 中文多字节序列：整词优先于逐字。
    #[test]
    fn cjk_multibyte_longest_match() {
        let t = tokenizer();
        assert_eq!(t.encode("北京".as_bytes()).unwrap(), vec![4]);
        assert_eq!(t.encode("上海".as_bytes()).unwrap(), vec![7]);
        assert_eq!(t.encode("海".as_bytes()).unwrap(), vec![6]);
        assert_eq!(t.encode("北京上海".as_bytes()).unwrap(), vec![4, 7]);
    }

    /// ★ choose 打分口径的命门：前导空格**原样保留**、由调用方负责，本分词器绝不做
    /// 任何归一化/注入。`" Paris"` 必须走空格前缀那条路（参照实现逐候选独立编码）。
    #[test]
    fn leading_space_is_preserved_not_normalized() {
        let t = tokenizer();
        assert_eq!(t.encode(b" Paris").unwrap(), vec![8]);
        assert_eq!(
            t.decode(&t.encode(b" Paris").unwrap()).unwrap(),
            &b" Paris"[..]
        );
        // 去掉空格就是另一个 token，证明没有隐含的前缀注入。
        assert_eq!(t.encode(b"Pari").unwrap(), vec![9]);
        // 反向也成立：词表里没有「单独一个空格」的词条，所以空格永远搭不了便车——
        // 前导空格只能靠带空格的那条词条本身命中。
        assert!(matches!(
            t.encode(b" Pari").unwrap_err(),
            TokenizerError::NoMatchingTokenFound
        ));
    }

    /// 字节数组形态：既支持非 UTF-8 裸字节，也支持与字符串等价的 UTF-8 序列。
    #[test]
    fn byte_array_vocab_entries() {
        let t = tokenizer();
        assert_eq!(t.encode(&[226, 130, 136]).unwrap(), vec![10]);
        assert_eq!(t.encode(&[128]).unwrap(), vec![13]);
        assert_eq!(t.decode(&[10, 13]).unwrap(), vec![226, 130, 136, 128]);
    }

    /// 孤立的 UTF-8 前导字节（长度 1 分支查不到 token）必须报错，而不是产出半个字。
    #[test]
    fn unencodable_bytes_error() {
        let t = tokenizer();
        for input in [&[0xe4u8][..], &[0xff][..], "上海x".as_bytes()] {
            let err = t.encode(input).unwrap_err();
            assert!(
                matches!(err, TokenizerError::NoMatchingTokenFound),
                "输入 {input:?} 必须报 NoMatchingTokenFound，实际：{err:?}"
            );
        }
    }

    /// 同一输入重复编码结果一致（choose 侧「同 payload 连打两次逐位一致」的前提）。
    #[test]
    fn encode_is_deterministic_and_appends() {
        let t = tokenizer();
        let payload = "上海北京".as_bytes();
        assert_eq!(t.encode(payload).unwrap(), t.encode(payload).unwrap());

        let mut output = t.encode(payload).unwrap();
        let before = output.clone();
        t.encode_into("北京".as_bytes(), &mut output).unwrap();
        assert_eq!(&output[..before.len()], &before[..]);
        assert_eq!(&output[before.len()..], &[4]);
    }

    #[test]
    fn decode_round_trips_mixed_text() {
        let t = tokenizer();
        let text = "北京上海 Paris猫ab";
        let tokens = t.encode(text.as_bytes()).unwrap();
        assert_eq!(tokens, vec![4, 7, 8, 12]);
        assert_eq!(t.decode(&tokens).unwrap(), text.as_bytes());
    }

    /// 解码越界的两个不同出口，钉住当前行为：超出最大 id 报错；词表里的 **id 空洞**
    /// 解码成空字节（不报错）——`token_index_to_bytes` 是按最大 id 扩容的，中间空洞
    /// 落在默认构造的空 `Vec` 上。
    #[test]
    fn decode_out_of_range_and_id_gap() {
        let t = tokenizer();
        let err = t.decode(&[21]).unwrap_err();
        assert!(
            matches!(err, TokenizerError::OutOfRangeToken(21)),
            "实际：{err:?}"
        );
        assert_eq!(t.decode(&[14]).unwrap(), Vec::<u8>::new());
    }
}
