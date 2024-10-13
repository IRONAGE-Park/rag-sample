use std::io::Write;

use pdf::{
    content::Op,
    enc::StreamFilter,
    file::FileOptions,
    object::{Resolve, XObject},
};

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
                Some(StreamFilter::CCITTFaxDecode(_)) => {
                    data = fax::tiff::wrap(&data, img.width, img.height).into();
                    "tiff"
                }
                _ => continue,
            };

            let fname = format!("./images/{}_extracted_image_{}.{}", name, i, ext);
            std::fs::write(fname.as_str(), data.clone()).unwrap();
        }

        println!("File pages {:?}", file.num_pages());
        file.pages().for_each(|page| {
            if page.is_err() {
                println!("Error: {:?}", page.err());
                return;
            }
            let page = page.unwrap();
            let contents = page.contents.as_ref().unwrap();
            let op = contents.operations(&resolver).unwrap();
            let content: String = op
                .iter()
                .map(|o| match o {
                    Op::TextDraw { text } => {
                        println!("Text: {:?}", text);
                        text.to_string_lossy()
                    }
                    Op::TextDrawAdjusted { array } => array
                        .iter()
                        .map(|p| p.to_string())
                        .collect::<Vec<String>>()
                        .join(" "),
                    _ => String::new(),
                })
                .collect();

            // println!("Contents: {:?}", content);
        });
    }
}

mod ocr;
mod pdf_reader;
mod text_store;
mod vector_store;
mod windows_api;
