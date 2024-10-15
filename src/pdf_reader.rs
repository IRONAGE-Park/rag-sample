use pdf::{content::Op, enc::StreamFilter, file::FileOptions, object::Resolve};

pub fn read_pdf_file(file_path: &std::path::Path) -> Result<String, String> {
    std::panic::catch_unwind(|| pdf_extract::extract_text(file_path).map_err(|e| e.to_string()))
        .unwrap_or_else(|_| Err("Catched panic by pdf_extract".to_string()))
}

pub fn read_pdf_by_pdf_rs(file_path: &std::path::Path) -> Result<String, String> {
    let file = FileOptions::cached()
        .open(file_path)
        .map_err(|e| e.to_string())?;
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
                .filter(|o| matches!(**o, pdf::object::XObject::Image(_))),
        );
    }

    let mut result = String::new();
    for (i, o) in images.iter().enumerate() {
        let img = match **o {
            pdf::object::XObject::Image(ref im) => im,
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

        // let fname = format!("./images/{}_extracted_image_{}.{}", name, i, ext);
        // std::fs::write(fname.as_str(), data.clone()).unwrap();
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

    Ok(result)
}

pub fn read_pdf_by_lopdf(file_path: &std::path::Path) -> Result<String, String> {
    let doc = lopdf::Document::load(file_path).map_err(|e| e.to_string())?;

    for (page_number, page_id) in doc.get_pages() {
        let page = doc
            .extract_text(&[page_number])
            .map_err(|op| op.to_string())?;
        println!("Page {}: {}", page_number, page);
        let page = page.replace(" \n", " ");
        std::fs::write(format!("./assets/page_{}.txt", page_number), page).unwrap();
    }

    Ok(String::new())
}
