use tantivy::{
    collector::TopDocs, doc, query::QueryParser, schema::*, DocAddress, Document, Index,
    IndexWriter, Score,
};

pub struct TantivyStore {
    schema: Schema,
    index: Index,
    index_writer: IndexWriter,

    fields: Fields,
}

impl TantivyStore {
    pub fn build() -> Result<Self, String> {
        let mut schema_builder = Schema::builder();

        let filepath = schema_builder.add_text_field("filepath", TEXT | STORED);
        let body = schema_builder.add_text_field("body", TEXT | STORED);
        let schema = schema_builder.build();

        let index = Index::open_in_dir("./index")
            .map_or_else(
                |_| Index::create_in_dir("./index", schema.clone()),
                |index| Ok(index),
            )
            .map_err(|e| e.to_string())?;
        let index_writer = index.writer(1_000_000_000).map_err(|e| e.to_string())?;

        Ok(Self {
            schema,
            index,
            index_writer,
            fields: Fields {
                file_path: filepath,
                body,
            },
        })
    }

    pub fn write_document(&mut self, path: &std::path::Path, text: String) -> Result<(), String> {
        let file_path = self.fields.file_path;
        let body = self.fields.body;

        self.index_writer
            .add_document(doc!(
                file_path => path.to_str().unwrap_or(""),
                body => text
            ))
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    pub fn commit(&mut self) -> Result<(), String> {
        self.index_writer
            .commit()
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    pub fn search(&self, query: &str) -> Result<Vec<(Score, String)>, String> {
        let reader = self.index.reader().map_err(|e| e.to_string())?;
        let searcher = reader.searcher();
        let query_parser =
            QueryParser::for_index(&self.index, vec![self.fields.file_path, self.fields.body]);

        let query = query_parser.parse_query(query).map_err(|e| e.to_string())?;

        let result: Vec<(Score, DocAddress)> = searcher
            .search(&query, &TopDocs::with_limit(10))
            .map_err(|e| e.to_string())?;

        let mut results = vec![];
        for (score, doc_address) in result {
            let retrieved_doc = searcher.doc::<TantivyDocument>(doc_address);
            if let Ok(retrieved_doc) = retrieved_doc {
                // let file_path = retrieved_doc
                //     .get_first(self.fields.file_path)
                //     .unwrap()
                //     .text()
                //     .unwrap();
                results.push((score, retrieved_doc.to_json(&self.schema)));
            }
        }

        Ok(results)
    }
}

struct Fields {
    file_path: Field,
    body: Field,
}
