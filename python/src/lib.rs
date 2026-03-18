use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use bm25rs::Method;

fn parse_method(method: &str) -> PyResult<Method> {
    match method.to_lowercase().as_str() {
        "lucene" => Ok(Method::Lucene),
        "robertson" => Ok(Method::Robertson),
        "atire" => Ok(Method::Atire),
        "bm25l" => Ok(Method::BM25L),
        "bm25+" | "bm25plus" => Ok(Method::BM25Plus),
        _ => Err(PyValueError::new_err(format!("Unknown method: {}", method))),
    }
}

#[pyclass(name = "BM25")]
struct PyBM25 {
    inner: bm25rs::BM25,
    index_path: Option<String>,
}

impl PyBM25 {
    fn auto_save(&self) -> PyResult<()> {
        if let Some(ref path) = self.index_path {
            self.inner
                .save(path)
                .map_err(|e| PyValueError::new_err(format!("Auto-save failed: {}", e)))?;
        }
        Ok(())
    }
}

#[pymethods]
impl PyBM25 {
    /// Create a new index.
    ///
    /// If `index` is provided, the index is persisted to that directory:
    /// - If the directory already contains a saved index, it is loaded automatically.
    /// - Every mutation (add, delete, update) auto-saves to disk.
    #[new]
    #[pyo3(signature = (index=None, method="lucene", k1=1.5, b=0.75, delta=0.5, use_stopwords=true))]
    fn new(
        index: Option<&str>,
        method: &str,
        k1: f32,
        b: f32,
        delta: f32,
        use_stopwords: bool,
    ) -> PyResult<Self> {
        if let Some(path) = index {
            let header = std::path::Path::new(path).join("header.bin");
            if header.exists() {
                let inner = bm25rs::BM25::load(path, true)
                    .map_err(|e| PyValueError::new_err(format!("Load failed: {}", e)))?;
                return Ok(PyBM25 {
                    inner,
                    index_path: Some(path.to_string()),
                });
            }
        }

        let m = parse_method(method)?;
        Ok(PyBM25 {
            inner: bm25rs::BM25::new(m, k1, b, delta, use_stopwords),
            index_path: index.map(|s| s.to_string()),
        })
    }

    /// Add documents to the index. Returns list of assigned indices.
    fn add(&mut self, documents: Vec<String>) -> PyResult<Vec<usize>> {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let ids = self.inner.add(&refs);
        self.auto_save()?;
        Ok(ids)
    }

    /// Search the index. Returns list of (index, score) tuples.
    /// If `subset` is provided, only those document IDs are scored (pre-filtering).
    #[pyo3(signature = (query, k, subset=None))]
    fn search(&self, query: &str, k: usize, subset: Option<Vec<usize>>) -> Vec<(usize, f32)> {
        let results = match subset {
            Some(ids) => self.inner.search_filtered(query, k, &ids),
            None => self.inner.search(query, k),
        };
        results.into_iter().map(|r| (r.index, r.score)).collect()
    }

    /// Delete documents by their indices.
    fn delete(&mut self, doc_ids: Vec<usize>) -> PyResult<()> {
        self.inner.delete(&doc_ids);
        self.auto_save()?;
        Ok(())
    }

    /// Update a document's text at the given index.
    fn update(&mut self, doc_id: usize, new_text: &str) -> PyResult<()> {
        self.inner.update(doc_id, new_text);
        self.auto_save()?;
        Ok(())
    }

    /// Save the index to a directory (explicit save, useful for in-memory indices).
    fn save(&self, index: &str) -> PyResult<()> {
        self.inner
            .save(index)
            .map_err(|e| PyValueError::new_err(format!("Save failed: {}", e)))
    }

    /// Load an index from a directory.
    #[staticmethod]
    #[pyo3(signature = (index, mmap=false))]
    fn load(index: &str, mmap: bool) -> PyResult<Self> {
        let inner = bm25rs::BM25::load(index, mmap)
            .map_err(|e| PyValueError::new_err(format!("Load failed: {}", e)))?;
        Ok(PyBM25 {
            inner,
            index_path: Some(index.to_string()),
        })
    }

    /// Number of active documents.
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pymodule]
fn bm25rs_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBM25>()?;
    Ok(())
}
