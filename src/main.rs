use std::{
    collections::HashMap,
    io::{Cursor, Write},
    time::SystemTime,
};

use futures_util::StreamExt;
// use langchain_rust::{
//     document_loaders::{pdf_extract_loader::PdfExtractLoader, Loader},
//     embedding::OllamaEmbedder,
//     schemas::Document as LangchainDocument,
//     vectorstore::{
//         qdrant::{Qdrant, StoreBuilder},
//         VecStoreOptions, VectorStore,
//     },
// };
use pdf::{
    content::Op,
    enc::StreamFilter,
    file::FileOptions,
    object::{Resolve, XObject},
    primitive::Primitive,
};
use rusty_tesseract::{image::ImageReader, Args, Image};
// use tantivy::{
//     collector::TopDocs, doc, query::QueryParser, schema::*, DocAddress, Document, Index,
//     IndexWriter, Score,
// };

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let now = SystemTime::now();

    let query = args.get(1).expect("No query provided");

    // let mut schema_builder = Schema::builder();

    // let filepath = schema_builder.add_text_field("filepath", TEXT | STORED);
    // let body = schema_builder.add_text_field("body", TEXT | STORED);
    // let schema = schema_builder.build();

    // let index = Index::open_in_dir("./index").unwrap_or_else(|_| {
    //     Index::create_in_dir("./index", schema.clone()).expect("Failed to create index")
    // });
    // let mut index_writer: IndexWriter = index
    //     .writer(1_000_000_000)
    //     .expect("Failed to create index writer");
    let files = windows_api::search_local_files_by_query(query.to_string())
        .expect("Failed to search files");
    println!("files {:?}", files.len());
    println!("질의문: {query}");
    // files.iter().for_each(|(name, path, size)| {
    //     println!("name: {}, size: {}", name, size);
    //     let path = std::path::Path::new(path.strip_prefix("file:").unwrap());
    //     let result =
    //         std::panic::catch_unwind(|| pdf_extract::extract_text(path).map_err(|e| e.to_string()))
    //             .unwrap_or_else(|e| {
    //                 println!("Failed to extract text from {}: {:?}", name, e);
    //                 Err("Catched panic by pdf_extract".to_string())
    //             });
    //     let result = result.map(|text| {
    //         index_writer
    //             .add_document(doc!(
    //                 filepath => path.to_str().unwrap_or(""),
    //                 body => text
    //             ))
    //             .expect("Failed to add document");
    //     });

    //     match result {
    //         Ok(_) => {
    //             println!("[SUCCESS]: {:?}", path);
    //         }
    //         Err(_) => {
    //             println!("[FAILURE] extract text from {}", name);
    //             return;
    //         }
    //     }
    // });
    println!("Committing {} files...", files.len());
    // index_writer.commit().expect("Failed to commit");

    // let reader = index.reader().expect("Failed to create reader");
    // let searcher = reader.searcher();
    // let query_parser = QueryParser::for_index(&index, vec![filepath, body]);

    // let query = query_parser
    //     .parse_query(query)
    //     .expect("Failed to parse query");

    // let top_docs: Vec<(Score, DocAddress)> = searcher
    //     .search(&query, &TopDocs::with_limit(10))
    //     .expect("Failed to search");

    // for (score, doc_address) in top_docs {
    //     let retrieved_doc = searcher
    //         .doc::<TantivyDocument>(doc_address)
    //         .expect("Failed to retrieve doc");
    //     println!("score: {}, {:?}", score, retrieved_doc.to_json(&schema));
    // }

    // if let Ok(elapsed) = now.elapsed() {
    //     println!(
    //         "Time: {}s",
    //         elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 * 1e-9
    //     );
    // }

    // let embedder = OllamaEmbedder::default().with_model("bge-m3");
    // let client = Qdrant::from_url("http://localhost:6334").build().unwrap();
    // client.delete_collection("langchain-rs").await.unwrap();
    // let store = StoreBuilder::new()
    //     .embedder(embedder)
    //     .client(client)
    //     .collection_name("langchain-rs")
    //     .build()
    //     .await
    //     .unwrap();

    for (name, path, size) in files.iter() {
        println!("name: {}, size: {}", name, size);
        let path = std::path::Path::new(path.strip_prefix("file:").unwrap());
        let file = FileOptions::cached().open(path).unwrap();
        let resolver = file.resolver();

        let mut images: Vec<_> = vec![];
        for page in file.pages() {
            if page.is_err() {
                continue;
            }
            let page = page.unwrap();
            let resources = page.resources().unwrap();
            images.extend(
                resources
                    .xobjects
                    .iter()
                    .map(|(_name, &r)| resolver.get(r).unwrap())
                    .filter(|o| matches!(**o, XObject::Image(_))),
            );
        }

        for (i, o) in images.iter().enumerate() {
            let img = match **o {
                XObject::Image(ref im) => im,
                _ => continue,
            };
            let (mut data, filter) = img
                .raw_image_data(&resolver)
                .expect("Failed to get image data");
            let ext = match filter {
                Some(StreamFilter::DCTDecode(_)) => "jpeg",
                Some(StreamFilter::JBIG2Decode(_)) => "jbig2",
                Some(StreamFilter::JPXDecode) => "jp2k",
                Some(StreamFilter::FlateDecode(_)) => "png",
                // Some(StreamFilter::CCITTFaxDecode(_)) => {
                //     data = fax::tiff::wrap(&data, img.width, img.height).into();
                //     "tiff"
                // }
                _ => continue,
            };

            let image_reader = ImageReader::new(Cursor::new(&data))
                .with_guessed_format()
                .unwrap();
            if image_reader.format().is_none() {
                continue;
            }
            let fname = format!("extracted_image_{}.{}", i, ext);
            println!("image: {:?}", fname);
            let img = image_reader.decode().unwrap();

            let args = Args {
                lang: "kor+eng".to_string(),
                config_variables: HashMap::new(),
                dpi: None,
                psm: None,
                oem: None,
            };
            let img = Image::from_dynamic_image(&img).expect("Failed to convert image");
            let output = rusty_tesseract::image_to_string(&img, &args).unwrap();
            // println!("Image {}: {}", i, output);
            std::fs::write(fname.as_str(), data.clone()).unwrap();
        }

        // println!("File pages {:?}", file.num_pages());
        // file.pages().for_each(|page| {
        //     let page = page.unwrap();
        //     let contents = page.contents.as_ref().unwrap();
        //     let op = contents.operations(&resolver).unwrap();
        //     let content: String = op
        //         .iter()
        //         .map(|o| match o {
        //             Op::TextDraw { text } => text.to_string_lossy(),
        //             Op::TextDrawAdjusted { array } => array
        //                 .iter()
        //                 .map(|p| p.to_string())
        //                 .collect::<Vec<String>>()
        //                 .join(" "),
        //             _ => String::new(),
        //         })
        //         .collect();

        //     println!("Contents: {:?}", content);
        // });

        // let loader = PdfExtractLoader::from_path(path).expect("Failed to load pdf");
        // let docs = loader
        //     .load()
        //     .await
        //     .unwrap()
        //     .map(|d| d.unwrap())
        //     .collect::<Vec<LangchainDocument>>()
        //     .await;

        // store
        //     .add_documents(&docs, &VecStoreOptions::default())
        //     .await
        //     .unwrap();
        // if let Ok(elapsed) = now.elapsed() {
        //     println!(
        //         "Time: {}s",
        //         elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 * 1e-9
        //     );
        // }
    }

    // print!("Query > ");
    // std::io::stdout().flush().unwrap();
    // let mut query = String::new();
    // std::io::stdin().read_line(&mut query).unwrap();

    // let results = store
    //     .similarity_search(&query, 3, &VecStoreOptions::default())
    //     .await
    //     .unwrap();

    // if results.is_empty() {
    //     println!("No results found.");
    //     return;
    // } else {
    //     results.iter().for_each(|r| {
    //         println!("Document: {:?}", r);
    //     });
    // }
}

mod windows_api;
