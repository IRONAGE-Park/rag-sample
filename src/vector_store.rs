use std::collections::HashMap;

use langchain_rust::{
    embedding::OllamaEmbedder,
    schemas::Document,
    vectorstore::{
        qdrant::{Qdrant, Store, StoreBuilder},
        VecStoreOptions, VectorStore,
    },
};
use serde_json::json;

pub struct QdrantStore {
    store: Store,
}

impl QdrantStore {
    pub async fn build() -> Result<Self, String> {
        let embedder = OllamaEmbedder::default().with_model("bge-m3");
        let client = Qdrant::from_url("http://localhost:6334")
            .build()
            .map_err(|e| e.to_string())?;
        let store = StoreBuilder::new()
            .embedder(embedder)
            .client(client)
            .collection_name("langchain-rs")
            .build()
            .await
            .map_err(|e| e.to_string())?;

        Ok(Self { store })
    }

    pub async fn write_document(
        &mut self,
        file_path: &std::path::Path,
        text: String,
    ) -> Result<(), String> {
        let document = Document::new(text).with_metadata(HashMap::from([(
            "path".to_string(),
            json!(file_path.to_str().unwrap_or("")),
        )]));

        self.store
            .add_documents(&[document], &VecStoreOptions::default())
            .await
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<Document>, String> {
        self.store
            .similarity_search(query, limit, &VecStoreOptions::default())
            .await
            .map_err(|e| e.to_string())
    }
}
