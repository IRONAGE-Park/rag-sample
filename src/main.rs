use std::{
    collections::HashMap,
    io::{Cursor, Write},
};

use futures_util::StreamExt;
use pdf::{
    content::Op,
    enc::StreamFilter,
    file::FileOptions,
    object::{Resolve, XObject},
    primitive::Primitive,
};
use rusty_tesseract::{image::ImageReader, Args, Image};

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
        .map(|(name, path, _)| (name, pdf_reader::read_pdf_file(path)))
        .collect();

    println!("Committing {} files...", files.len());

    print!("Contents Query > ");
    std::io::stdout().flush().unwrap();
    let mut query = String::new();
    std::io::stdin().read_line(&mut query).unwrap();

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
    }
}

mod pdf_reader;
mod text_store;
mod vector_store;
mod windows_api;
