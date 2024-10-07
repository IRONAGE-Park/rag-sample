use std::time::SystemTime;

use tantivy::{
    collector::TopDocs, doc, query::QueryParser, schema::*, DocAddress, Index, IndexWriter, Score,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let now = SystemTime::now();

    let query = args.get(1).expect("No query provided");

    let mut schema_builder = Schema::builder();

    let filepath = schema_builder.add_text_field("filepath", TEXT | STORED);
    let body = schema_builder.add_text_field("body", TEXT | STORED);
    let schema = schema_builder.build();

    // let index = Index::create_in_dir("./index", schema.clone()).expect("Failed to create index");
    let index = Index::open_in_dir("./index").expect("Failed to open index");
    let mut index_writer: IndexWriter = index
        .writer(1_000_000_000)
        .expect("Failed to create index writer");
    let files = windows_api::search_local_files_by_query(query.to_string())
        .expect("Failed to search files");
    println!("files {:?}", files.len());
    println!("질의문: {query}");
    files.iter().for_each(|(name, path, size)| {
        println!("name: {}, size: {}", name, size);
        let path = std::path::Path::new(path.strip_prefix("file:").unwrap());
        let result =
            std::panic::catch_unwind(|| pdf_extract::extract_text(path).map_err(|e| e.to_string()))
                .unwrap_or_else(|e| {
                    println!("Failed to extract text from {}: {:?}", name, e);
                    Err("Catched panic by pdf_extract".to_string())
                });
        let result = result.map(|text| {
            index_writer
                .add_document(doc!(
                    filepath => path.to_str().unwrap_or(""),
                    body => text
                ))
                .expect("Failed to add document");
        });

        match result {
            Ok(_) => {
                println!("[SUCCESS]: {:?}", path);
            }
            Err(_) => {
                println!("[FAILURE] extract text from {}", name);
                return;
            }
        }
    });
    println!("Committing {} files...", files.len());
    index_writer.commit().expect("Failed to commit");

    let reader = index.reader().expect("Failed to create reader");
    let searcher = reader.searcher();
    let query_parser = QueryParser::for_index(&index, vec![filepath, body]);

    let query = query_parser
        .parse_query(query)
        .expect("Failed to parse query");

    let top_docs: Vec<(Score, DocAddress)> = searcher
        .search(&query, &TopDocs::with_limit(10))
        .expect("Failed to search");

    for (score, doc_address) in top_docs {
        let retrieved_doc = searcher
            .doc::<TantivyDocument>(doc_address)
            .expect("Failed to retrieve doc");
        println!("score: {}, {:?}", score, retrieved_doc.to_json(&schema));
    }

    if let Ok(elapsed) = now.elapsed() {
        println!(
            "Time: {}s",
            elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 * 1e-9
        );
    }
}

mod windows_api;
