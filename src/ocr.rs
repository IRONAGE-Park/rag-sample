use std::{collections::HashMap, io::Cursor};

use rusty_tesseract::{image::ImageReader, Args, Image};

// pub fn ocr_by_path(file_path: &std::path::Path) -> String {
//     let img = image_reader.decode().unwrap();

//     let args = Args {
//         lang: "kor+eng".to_string(),
//         config_variables: HashMap::new(),
//         dpi: None,
//         psm: None,
//         oem: None,
//     };
//     let img = Image::from_dynamic_image(&img).expect("Failed to convert image");
//     rusty_tesseract::image_to_string(&img, &args).unwrap_or(String::new())
// }

pub fn ocr_by_buffer(data: &[u8]) -> String {
    let image_reader = ImageReader::new(Cursor::new(&data))
        .with_guessed_format()
        .unwrap();
    if image_reader.format().is_none() {
        return String::new();
    }
    let img = image_reader.decode().unwrap();

    let args = Args {
        lang: "kor+eng".to_string(),
        config_variables: HashMap::new(),
        dpi: None,
        psm: None,
        oem: None,
    };
    let img = Image::from_dynamic_image(&img).expect("Failed to convert image");
    rusty_tesseract::image_to_string(&img, &args).unwrap_or(String::new())
}
