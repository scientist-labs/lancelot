use magnus::{Error, Ruby, RHash, Symbol, Value, TryConvert, r_hash::ForEach};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use std::sync::Arc;

pub fn build_arrow_schema(ruby: &Ruby, schema_hash: RHash) -> Result<ArrowSchema, Error> {
    let mut fields = Vec::new();

    schema_hash.foreach(|key: Symbol, value: Value| {
        let field_name = key.name()?.to_string();

        let data_type = if let Some(hash) = RHash::from_value(value) {
            let type_str: String = hash.fetch(ruby.to_symbol("type"))?;

            match type_str.as_str() {
                "vector" => {
                    let dimension: i32 = hash.fetch(ruby.to_symbol("dimension"))?;
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dimension,
                    )
                }
                _ => return Err(Error::new(
                    ruby.exception_arg_error(),
                    format!("Unknown field type: {}", type_str)
                ))
            }
        } else {
            let type_str = String::try_convert(value)?;
            match type_str.as_str() {
                "string" => DataType::Utf8,
                "float32" => DataType::Float32,
                "float64" => DataType::Float64,
                "int32" => DataType::Int32,
                "int64" => DataType::Int64,
                "boolean" => DataType::Boolean,
                _ => return Err(Error::new(
                    ruby.exception_arg_error(),
                    format!("Unknown field type: {}", type_str)
                ))
            }
        };

        fields.push(Field::new(field_name, data_type, true));
        Ok(ForEach::Continue)
    })?;

    Ok(ArrowSchema::new(fields))
}
