use std::io::Write;

use pyo3::{
    types::{PyAnyMethods, PyModule, PyTuple},
    Py, PyAny, PyResult, Python,
};

#[tokio::main]
async fn main() {
    // 1. 쿼리 가져오기
    let args: Vec<String> = std::env::args().collect();
    let query = args.get(1).map_or(String::from("."), |v| v.to_string());
    println!("File Query: {query} (Default: .)");

    // 2. 쿼리에 해당하는 파일들을 가져오기
    let files = native::search_local_files_by_query(query.to_string())
        .expect("Failed to search files");
    let files = files
        .iter()
        // 2-1. 존재하는 파일만 필터링
        .filter(|(_, path, _)| {
            std::path::Path::new(path.strip_prefix("file:").unwrap_or(path)).exists()
        })
        .collect::<Vec<_>>();

    // 3. 이미지 캡션 모델 사전 로드
    if let Some((_, path, _)) = files
        .iter()
        .filter(|(_, path, _)| path.ends_with(".jpg") || path.ends_with(".png"))
        .collect::<Vec<_>>()
        .first()
    {
        let _: PyResult<()> = Python::with_gil(move |py| {
            let initalize_module = PyModule::from_code_bound(
                py,
                include_str!("../src-python/core/blip/blip_model.py"),
                "blip_model.py",
                "blip_model",
            )?;
            let initialize_vision_model: Py<PyAny> =
                initalize_module.getattr("initialize_vision_model")?.into();
            let initialize_text_decoder: Py<PyAny> =
                initalize_module.getattr("initialize_text_decoder")?.into();
    
            let path = path.strip_prefix("file:").unwrap_or(path);
    
            let arg = PyTuple::new_bound(py, &[path]);
            initialize_vision_model.call1(py, arg)?;
            initialize_text_decoder.call0(py)?;
            Ok(())
        });
    }
    
    let time = std::time::SystemTime::now();
    // 4. 파일들을 `Python`으로 전달하여 `Vector Store`에 저장
    let success_length: PyResult<usize> = Python::with_gil(|py| {
        PyModule::from_code_bound(
            py,
            include_str!("../src-python/core/blip/blip_model.py"),
            "blip_model.py",
            "core.blip.blip_model",
        )?;
        PyModule::from_code_bound(
            py,
            include_str!("../src-python/core/blip/blip.py"),
            "blip.py",
            "core.blip.blip",
        )?;
        PyModule::from_code_bound(
            py,
            include_str!("../src-python/core/loader/pdf.py"),
            "pdf.py",
            "core.loader.pdf",
        )?;
        PyModule::from_code_bound(
            py,
            include_str!("../src-python/core/loader/image.py"),
            "image.py",
            "core.loader.image",
        )?;
        PyModule::from_code_bound(
            py,
            include_str!("../src-python/core/vector_store.py"),
            "vector_store.py",
            "core.vector_store",
        )?;
        let module = PyModule::from_code_bound(
            py,
            include_str!("../src-python/main.py"),
            "main.py",
            "main",
        )?;
        let pdf_embed_func: Py<PyAny> = module.getattr("pdf_embed")?.into();
        let image_embed_func: Py<PyAny> = module.getattr("image_embed")?.into();
    
        let success_list = files
            .iter()
            .filter(|(name, path, _)| {
                let path = path.strip_prefix("file:").unwrap_or(path);
                let arg = PyTuple::new_bound(py, &[path]);
                println!("Parsing {}", name);
                if path.ends_with(".pdf") {
                    pdf_embed_func.call1(py, arg)
                } else {
                    image_embed_func.call1(py, arg)
                }
                .is_ok()
            })
            .collect::<Vec<_>>();
    
        Ok(success_list.len())
    });

    if let Ok(duration) = time.elapsed() {
        println!("Elapsed Time: {:?}", duration);
    }

    println!(
        "Committing {} files...",
        success_length.expect("Failed to embed files")
    );

    // 5. `Vector Store`에 저장된 파일들을 검색
    print!("Contents Query > ");
    std::io::stdout().flush().unwrap();
    let mut query = String::new();
    std::io::stdin().read_line(&mut query).unwrap();
}

mod native;
// mod ocr;
// mod pdf_reader;
// mod text_store;
// mod vector_store;
