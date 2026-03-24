use magnus::{Error, Ruby, Module};

mod dataset;
mod schema;
mod conversion;

use dataset::LancelotDataset;

#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let module = ruby.define_module("Lancelot")?;

    let dataset_class = module.define_class("Dataset", ruby.class_object())?;
    LancelotDataset::bind(&dataset_class)?;

    Ok(())
}
