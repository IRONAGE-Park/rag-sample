use std::path::Path;

use image::ImageReader;
use openvino::{
    prepostprocess, Core, DeviceType, ElementType, Layout, ResizeAlgorithm, Shape, Tensor,
};

fn main() {
    let path = Path::new(".");
    let path = path.join("python").join("blip");
    let model_path = path.join("blip_vision_model.xml");
    let weights_path = path.join("blip_vision_model.bin");
    println!("model_path: {:?}", model_path);

    let image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png";
    let response = reqwest::blocking::get(image_url).unwrap();
    let response = response.bytes().unwrap();
    let cursor = std::io::Cursor::new(response);
    let image = ImageReader::new(cursor)
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap();

    let mut core = Core::new().unwrap();
    let mut vision_model = core
        .read_model_from_file(model_path.to_str().unwrap(), weights_path.to_str().unwrap())
        .unwrap();

    // vision_model.
    let model = core.compile_model(&vision_model, DeviceType::GPU).unwrap();

    let shape = Shape::new(&[1, 3, 224, 224]).unwrap();
    let layout = Layout::new("NCHW").unwrap();
    let element = ElementType::F32;
    let mut tensor = Tensor::new(element, &shape).unwrap();
}
