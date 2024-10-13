use std::io::Write;

#[tokio::main]
async fn main() {
    // 1. 쿼리 가져오기
    let args: Vec<String> = std::env::args().collect();
    let query = args.get(1).map_or(String::from("."), |v| v.to_string());

    println!("File Query: {query} (Default: .)");
    // 쿼리에 해당하는 파일들을 가져옴
    let files = windows_api::search_local_files_by_query(query.to_string())
        .expect("Failed to search files");

    let pdf_files: Vec<(&String, Result<String, String>)> = files
        .iter()
        .filter(|(name, _, _)| name.ends_with(".pdf"))
        .map(|(name, path, _)| (name, pdf_reader::read_pdf_file(std::path::Path::new(path))))
        .collect();

    files.iter().for_each(|(name, path, _)| {
        println!("File: {} - {}", name, path);
        let path = path.strip_prefix("file:").unwrap_or(path);
        pdf_reader::read_pdf_by_lopdf(std::path::Path::new(path));
    });

    println!("Committing {} files...", files.len());

    print!("Contents Query > ");
    std::io::stdout().flush().unwrap();
    let mut query = String::new();
    std::io::stdin().read_line(&mut query).unwrap();
}

mod ocr;
mod pdf_reader;
mod text_store;
mod vector_store;
mod windows_api;
