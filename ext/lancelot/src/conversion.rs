use magnus::{Error, Ruby, RHash, RArray, Value, TryConvert, value::ReprValue};
use arrow_schema::{DataType, Schema as ArrowSchema};
use arrow_array::{RecordBatch, StringArray, Float32Array, ArrayRef, Array, FixedSizeListArray};
use std::collections::HashMap;
use std::sync::Arc;

pub fn build_record_batch(
    ruby: &Ruby,
    data: RArray,
    schema: &ArrowSchema,
) -> Result<RecordBatch, Error> {
    let mut columns: HashMap<String, Vec<Option<String>>> = HashMap::new();
    let mut float_columns: HashMap<String, Vec<Option<f32>>> = HashMap::new();
    let mut int_columns: HashMap<String, Vec<Option<i64>>> = HashMap::new();
    let mut bool_columns: HashMap<String, Vec<Option<bool>>> = HashMap::new();
    let mut vector_columns: HashMap<String, Vec<Option<Vec<f32>>>> = HashMap::new();

    for field in schema.fields() {
        match field.data_type() {
            DataType::Utf8 => {
                columns.insert(field.name().to_string(), Vec::new());
            }
            DataType::Float32 => {
                float_columns.insert(field.name().to_string(), Vec::new());
            }
            DataType::Int64 => {
                int_columns.insert(field.name().to_string(), Vec::new());
            }
            DataType::Boolean => {
                bool_columns.insert(field.name().to_string(), Vec::new());
            }
            DataType::FixedSizeList(_, _) => {
                vector_columns.insert(field.name().to_string(), Vec::new());
            }
            _ => {}
        }
    }

    // Index-based iteration over data RArray
    for idx in 0..data.len() {
        let item_value: Value = data.entry(idx as isize)?;
        let item = RHash::try_convert(item_value)?;
        for field in schema.fields() {
            let key = ruby.to_symbol(field.name());
            // Make fields optional - use get instead of fetch
            let value: Value = item.get(key)
                .or_else(|| {
                    // Try with string key
                    item.get(field.name().as_str())
                })
                .unwrap_or_else(|| ruby.qnil().as_value());

            match field.data_type() {
                DataType::Utf8 => {
                    if value.is_nil() {
                        columns.get_mut(field.name()).unwrap().push(None);
                    } else {
                        let s = String::try_convert(value)?;
                        columns.get_mut(field.name()).unwrap().push(Some(s));
                    }
                }
                DataType::Float32 => {
                    if value.is_nil() {
                        float_columns.get_mut(field.name()).unwrap().push(None);
                    } else {
                        let f = f64::try_convert(value)?;
                        float_columns.get_mut(field.name()).unwrap().push(Some(f as f32));
                    }
                }
                DataType::Int64 => {
                    if value.is_nil() {
                        int_columns.get_mut(field.name()).unwrap().push(None);
                    } else {
                        let i = i64::try_convert(value)?;
                        int_columns.get_mut(field.name()).unwrap().push(Some(i));
                    }
                }
                DataType::Boolean => {
                    if value.is_nil() {
                        bool_columns.get_mut(field.name()).unwrap().push(None);
                    } else {
                        let b = bool::try_convert(value)?;
                        bool_columns.get_mut(field.name()).unwrap().push(Some(b));
                    }
                }
                DataType::FixedSizeList(_, _) => {
                    if value.is_nil() {
                        vector_columns.get_mut(field.name()).unwrap().push(None);
                    } else {
                        let arr = RArray::try_convert(value)?;
                        let len = arr.len();
                        let mut vec: Vec<f32> = Vec::with_capacity(len);
                        for j in 0..len {
                            let v: f64 = arr.entry(j as isize)?;
                            vec.push(v as f32);
                        }
                        vector_columns.get_mut(field.name()).unwrap().push(Some(vec));
                    }
                }
                _ => {}
            }
        }
    }

    let mut arrays: Vec<ArrayRef> = Vec::new();

    for field in schema.fields() {
        let array: ArrayRef = match field.data_type() {
            DataType::Utf8 => {
                let values = columns.get(field.name()).unwrap();
                Arc::new(StringArray::from(values.clone()))
            }
            DataType::Float32 => {
                let values = float_columns.get(field.name()).unwrap();
                Arc::new(Float32Array::from(values.clone()))
            }
            DataType::Int64 => {
                let values = int_columns.get(field.name()).unwrap();
                Arc::new(arrow_array::Int64Array::from(values.clone()))
            }
            DataType::Boolean => {
                let values = bool_columns.get(field.name()).unwrap();
                Arc::new(arrow_array::BooleanArray::from(values.clone()))
            }
            DataType::FixedSizeList(inner_field, list_size) => {
                let values = vector_columns.get(field.name()).unwrap();
                // Build flat array of all values
                let mut flat_values = Vec::new();
                for vec_opt in values {
                    match vec_opt {
                        Some(vec) => {
                            if vec.len() != *list_size as usize {
                                return Err(Error::new(
                                    ruby.exception_arg_error(),
                                    format!("Vector dimension mismatch. Expected {}, got {}", list_size, vec.len())
                                ));
                            }
                            flat_values.extend(vec);
                        }
                        None => {
                            // Add nulls for the entire vector
                            flat_values.extend(vec![0.0f32; *list_size as usize]);
                        }
                    }
                }

                let flat_array = Float32Array::from(flat_values);
                Arc::new(FixedSizeListArray::new(
                    inner_field.clone(),
                    *list_size,
                    Arc::new(flat_array),
                    None
                ))
            }
            _ => return Err(Error::new(
                ruby.exception_runtime_error(),
                format!("Unsupported data type: {:?}", field.data_type())
            ))
        };

        arrays.push(array);
    }

    RecordBatch::try_new(Arc::new(schema.clone()), arrays)
        .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))
}

pub fn convert_batch_to_ruby(ruby: &Ruby, batch: &RecordBatch) -> Result<RArray, Error> {
    let documents = ruby.ary_new();

    let num_rows = batch.num_rows();
    let schema = batch.schema();

    for row_idx in 0..num_rows {
        let doc = ruby.hash_new();

        for (col_idx, field) in schema.fields().iter().enumerate() {
            let column = batch.column(col_idx);
            let key = ruby.to_symbol(field.name());

            // CRITICAL: Add bounds checking for all array access
            if row_idx >= column.len() {
                return Err(Error::new(
                    ruby.exception_runtime_error(),
                    format!("Row index {} out of bounds for column '{}' with length {}",
                            row_idx, field.name(), column.len())
                ));
            }

            match field.data_type() {
                DataType::Utf8 => {
                    let array = column.as_any().downcast_ref::<StringArray>()
                        .ok_or_else(|| Error::new(ruby.exception_runtime_error(), "Failed to cast to StringArray"))?;

                    if array.is_null(row_idx) {
                        doc.aset(key, ruby.qnil())?;
                    } else {
                        doc.aset(key, array.value(row_idx))?;
                    }
                }
                DataType::Float32 => {
                    let array = column.as_any().downcast_ref::<Float32Array>()
                        .ok_or_else(|| Error::new(ruby.exception_runtime_error(), "Failed to cast to Float32Array"))?;

                    if array.is_null(row_idx) {
                        doc.aset(key, ruby.qnil())?;
                    } else {
                        doc.aset(key, array.value(row_idx))?;
                    }
                }
                DataType::Int64 => {
                    let array = column.as_any().downcast_ref::<arrow_array::Int64Array>()
                        .ok_or_else(|| Error::new(ruby.exception_runtime_error(), "Failed to cast to Int64Array"))?;

                    if array.is_null(row_idx) {
                        doc.aset(key, ruby.qnil())?;
                    } else {
                        doc.aset(key, array.value(row_idx))?;
                    }
                }
                DataType::Boolean => {
                    let array = column.as_any().downcast_ref::<arrow_array::BooleanArray>()
                        .ok_or_else(|| Error::new(ruby.exception_runtime_error(), "Failed to cast to BooleanArray"))?;

                    if array.is_null(row_idx) {
                        doc.aset(key, ruby.qnil())?;
                    } else {
                        doc.aset(key, array.value(row_idx))?;
                    }
                }
                DataType::FixedSizeList(_, list_size) => {
                    let array = column.as_any().downcast_ref::<FixedSizeListArray>()
                        .ok_or_else(|| Error::new(ruby.exception_runtime_error(), "Failed to cast to FixedSizeListArray"))?;

                    if array.is_null(row_idx) {
                        doc.aset(key, ruby.qnil())?;
                    } else {
                        let values = array.value(row_idx);
                        let float_array = values.as_any().downcast_ref::<Float32Array>()
                            .ok_or_else(|| Error::new(ruby.exception_runtime_error(), "Failed to cast vector values to Float32Array"))?;

                        // CRITICAL: Verify the float_array has the expected size
                        let expected_size = *list_size as usize;
                        if float_array.len() != expected_size {
                            return Err(Error::new(
                                ruby.exception_runtime_error(),
                                format!("Vector data corruption: expected {} elements but found {} for field '{}'",
                                        expected_size, float_array.len(), field.name())
                            ));
                        }

                        let ruby_array = ruby.ary_new();
                        for i in 0..expected_size {
                            ruby_array.push(float_array.value(i))?;
                        }
                        doc.aset(key, ruby_array)?;
                    }
                }
                _ => {
                    // Skip unsupported types for now
                }
            }
        }

        documents.push(doc)?;
    }

    Ok(documents)
}
