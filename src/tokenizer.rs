use rust_stemmers::{Algorithm, Stemmer};
use unicode_normalization::UnicodeNormalization;

/// Tokenizer mode controls text preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerMode {
    /// Lowercase + split on non-alphanumeric. Fast, no stemming.
    Plain,
    /// Plain + Unicode NFKD normalization (diacritics folded to ASCII).
    Unicode,
    /// Plain + Snowball stemming (English).
    Stem,
    /// Unicode normalization + Snowball stemming. Best accuracy.
    UnicodeStem,
}

/// A configurable tokenizer: lowercase, split, optional unicode folding,
/// optional stemming, optional stopword removal.
pub struct Tokenizer {
    stopwords: Option<std::collections::HashSet<String>>,
    mode: TokenizerMode,
    stemmer: Option<Stemmer>,
}

const ENGLISH_STOPWORDS: &[&str] = &[
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "its",
    "itself",
    "them",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "am",
    "been",
    "being",
    "do",
    "does",
    "did",
    "doing",
    "would",
    "should",
    "could",
    "ought",
    "might",
    "shall",
    "can",
    "need",
    "dare",
    "had",
    "has",
    "have",
    "having",
    "about",
    "above",
    "after",
    "again",
    "against",
    "below",
    "between",
    "during",
    "from",
    "further",
    "here",
    "once",
    "only",
    "out",
    "over",
    "same",
    "so",
    "than",
    "too",
    "under",
    "until",
    "up",
    "very",
    "own",
    "just",
    "don",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "s",
    "t",
    "ve",
];

impl Tokenizer {
    /// Create a tokenizer. For backward compatibility, `new(use_stopwords)` uses `Plain` mode.
    pub fn new(use_stopwords: bool) -> Self {
        Self::with_mode(TokenizerMode::Plain, use_stopwords)
    }

    /// Create a tokenizer with a specific mode.
    pub fn with_mode(mode: TokenizerMode, use_stopwords: bool) -> Self {
        let stopwords = if use_stopwords {
            Some(ENGLISH_STOPWORDS.iter().map(|s| s.to_string()).collect())
        } else {
            None
        };
        let stemmer = match mode {
            TokenizerMode::Stem | TokenizerMode::UnicodeStem => {
                Some(Stemmer::create(Algorithm::English))
            }
            _ => None,
        };
        Tokenizer {
            stopwords,
            mode,
            stemmer,
        }
    }

    /// Tokenize and return owned lowercase (+ optionally stemmed/normalized) tokens.
    pub fn tokenize_owned(&self, text: &str) -> Vec<String> {
        // Step 1: Lowercase
        let lowered = text.to_lowercase();

        // Step 2: Optional Unicode NFKD normalization → strip diacritics
        let normalized: String = match self.mode {
            TokenizerMode::Unicode | TokenizerMode::UnicodeStem => fold_to_ascii(&lowered),
            _ => lowered,
        };

        // Step 3: Split on non-alphanumeric
        let raw_tokens = split_alphanumeric(&normalized);

        // Step 4: Stopword filter + optional stemming
        let mut tokens = Vec::with_capacity(raw_tokens.len());
        for token in raw_tokens {
            if !self.should_keep(&token) {
                continue;
            }
            let final_token = if let Some(ref stemmer) = self.stemmer {
                stemmer.stem(&token).into_owned()
            } else {
                token
            };
            if !final_token.is_empty() {
                tokens.push(final_token);
            }
        }
        tokens
    }

    #[inline]
    fn should_keep(&self, token: &str) -> bool {
        if token.is_empty() {
            return false;
        }
        if let Some(ref sw) = self.stopwords {
            !sw.contains(token)
        } else {
            true
        }
    }
}

/// Split a string on non-alphanumeric boundaries, returning owned tokens.
fn split_alphanumeric(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut start = None;

    for (i, c) in text.char_indices() {
        if c.is_alphanumeric() {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start {
            tokens.push(text[s..i].to_string());
            start = None;
        }
    }
    if let Some(s) = start {
        tokens.push(text[s..].to_string());
    }
    tokens
}

/// NFKD normalize then strip non-ASCII (removes diacritics: "café" → "cafe").
fn fold_to_ascii(text: &str) -> String {
    text.nfkd().filter(|c| c.is_ascii()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plain_basic() {
        let tok = Tokenizer::new(false);
        assert_eq!(tok.tokenize_owned("Hello WORLD"), vec!["hello", "world"]);
    }

    #[test]
    fn test_plain_stopwords() {
        let tok = Tokenizer::new(true);
        let tokens = tok.tokenize_owned("the quick brown fox");
        assert_eq!(tokens, vec!["quick", "brown", "fox"]);
    }

    #[test]
    fn test_plain_punctuation() {
        let tok = Tokenizer::new(false);
        let tokens = tok.tokenize_owned("hello, world! how's it going?");
        assert_eq!(tokens, vec!["hello", "world", "how", "s", "it", "going"]);
    }

    #[test]
    fn test_unicode_diacritics() {
        let tok = Tokenizer::with_mode(TokenizerMode::Unicode, false);
        let tokens = tok.tokenize_owned("café résumé naïve");
        assert_eq!(tokens, vec!["cafe", "resume", "naive"]);
    }

    #[test]
    fn test_unicode_fullwidth() {
        let tok = Tokenizer::with_mode(TokenizerMode::Unicode, false);
        // NFKD normalizes fullwidth chars
        let tokens = tok.tokenize_owned("Ｈｅｌｌｏ");
        assert_eq!(tokens, vec!["hello"]);
    }

    #[test]
    fn test_stem_basic() {
        let tok = Tokenizer::with_mode(TokenizerMode::Stem, false);
        let tokens = tok.tokenize_owned("running cancellation connections");
        assert_eq!(tokens, vec!["run", "cancel", "connect"]);
    }

    #[test]
    fn test_stem_with_stopwords() {
        let tok = Tokenizer::with_mode(TokenizerMode::Stem, true);
        let tokens = tok.tokenize_owned("the cats are running quickly");
        // "the" and "are" removed, then stemmed
        // Snowball English: "cats"→"cat", "running"→"run", "quickly"→"quick"
        assert_eq!(tokens, vec!["cat", "run", "quick"]);
    }

    #[test]
    fn test_unicode_stem() {
        let tok = Tokenizer::with_mode(TokenizerMode::UnicodeStem, false);
        let tokens = tok.tokenize_owned("café résumés naïvely");
        assert_eq!(tokens, vec!["cafe", "resum", "naiv"]);
    }

    #[test]
    fn test_unicode_lowercase() {
        let tok = Tokenizer::with_mode(TokenizerMode::Unicode, false);
        // Full Unicode lowercase (not just ASCII)
        let tokens = tok.tokenize_owned("Ünïcödé STRASSE");
        assert_eq!(tokens, vec!["unicode", "strasse"]);
    }

    #[test]
    fn test_modes_different_output() {
        let text = "The résumés are running";
        let plain = Tokenizer::with_mode(TokenizerMode::Plain, true).tokenize_owned(text);
        let unicode = Tokenizer::with_mode(TokenizerMode::Unicode, true).tokenize_owned(text);
        let stem = Tokenizer::with_mode(TokenizerMode::Stem, true).tokenize_owned(text);
        let us = Tokenizer::with_mode(TokenizerMode::UnicodeStem, true).tokenize_owned(text);

        // Plain keeps diacritics
        assert!(plain.contains(&"résumés".to_string()));
        // Unicode folds them
        assert!(unicode.contains(&"resumes".to_string()));
        // Stem stems but keeps diacritics
        assert!(stem.iter().any(|t| t.contains("résum")));
        // UnicodeStem folds + stems
        assert!(us.contains(&"resum".to_string()));
    }
}
