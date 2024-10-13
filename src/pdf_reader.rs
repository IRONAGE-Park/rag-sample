pub fn read_pdf_file(file_path: &String) -> Result<String, String> {
    std::panic::catch_unwind(|| pdf_extract::extract_text(file_path).map_err(|e| e.to_string()))
        .unwrap_or_else(|_| Err("Catched panic by pdf_extract".to_string()))
}
