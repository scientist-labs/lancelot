use magnus::{Error, Ruby, RHash, RArray, Value, function, method, RClass, Module, Object};
use std::cell::RefCell;
use std::sync::Arc;
use tokio::runtime::Runtime;
use lance::Dataset;
use lance::index::vector::VectorIndexParams;
use lance_index::{IndexType, DatasetIndexExt};
use lance_index::scalar::{InvertedIndexParams, FullTextSearchQuery};
use arrow_array::{RecordBatch, RecordBatchIterator, Float32Array};
use futures::stream::TryStreamExt;

use crate::schema::build_arrow_schema;
use crate::conversion::{build_record_batch, convert_batch_to_ruby};
use arrow_schema::DataType;

/// Convert Arrow DataType to Ruby-friendly string representation
fn datatype_to_ruby_string(dtype: &DataType) -> &'static str {
    match dtype {
        DataType::Utf8 | DataType::LargeUtf8 => "string",
        DataType::Boolean => "boolean",
        DataType::Int8 | DataType::Int16 | DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 => "int32",
        DataType::UInt64 => "int64",
        DataType::Float32 => "float32",
        DataType::Float64 => "float64",
        DataType::Date32 => "date",
        DataType::Date64 => "datetime",
        DataType::Timestamp(_, _) => "datetime",
        DataType::FixedSizeList(_, _) => "vector", // Will be handled specially
        _ => "unknown"
    }
}

#[magnus::wrap(class = "Lancelot::Dataset", free_immediately, size)]
pub struct LancelotDataset {
    dataset: RefCell<Option<Dataset>>,
    runtime: RefCell<Runtime>,
    path: String,
}

impl LancelotDataset {
    pub fn new(path: String) -> Result<Self, Error> {
        let ruby = Ruby::get().unwrap();
        let runtime = Runtime::new()
            .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;

        Ok(Self {
            dataset: RefCell::new(None),
            runtime: RefCell::new(runtime),
            path,
        })
    }

    pub fn path(&self) -> String {
        self.path.clone()
    }

    pub fn create(&self, schema_hash: RHash) -> Result<(), Error> {
        let ruby = Ruby::get().unwrap();
        let schema = build_arrow_schema(&ruby, schema_hash)?;

        let empty_batch = RecordBatch::new_empty(Arc::new(schema.clone()));
        let batches = vec![empty_batch];
        let reader = RecordBatchIterator::new(
            batches.into_iter().map(Ok),
            Arc::new(schema)
        );

        let dataset = self.runtime.borrow_mut().block_on(async {
            Dataset::write(
                reader,
                &self.path,
                None,
            )
            .await
            .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))
        })?;

        self.dataset.replace(Some(dataset));
        Ok(())
    }

    pub fn open(&self) -> Result<(), Error> {
        let ruby = Ruby::get().unwrap();
        let dataset = self.runtime.borrow_mut().block_on(async {
            Dataset::open(&self.path)
                .await
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))
        })?;

        self.dataset.replace(Some(dataset));
        Ok(())
    }

    pub fn add_data(&self, data: RArray) -> Result<(), Error> {
        let ruby = Ruby::get().unwrap();
        let mut dataset = self.dataset.borrow_mut();
        let dataset = dataset.as_mut()
            .ok_or_else(|| Error::new(ruby.exception_runtime_error(), "Dataset not opened"))?;

        // Check if data is empty
        if data.len() == 0 {
            return Ok(());  // Nothing to add
        }

        // Get the dataset's schema
        let schema = self.runtime.borrow_mut().block_on(async {
            dataset.schema()
        });

        // Convert Lance schema to Arrow schema
        let arrow_schema = schema.into();

        let batch = build_record_batch(&ruby, data, &arrow_schema)?;

        let batches = vec![batch];
        let reader = RecordBatchIterator::new(
            batches.into_iter().map(Ok),
            Arc::new(arrow_schema)
        );

        self.runtime.borrow_mut().block_on(async move {
            dataset.append(reader, None)
                .await
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))
        })?;

        Ok(())
    }

    pub fn count_rows(&self) -> Result<i64, Error> {
        let ruby = Ruby::get().unwrap();
        let dataset = self.dataset.borrow();
        let dataset = dataset.as_ref()
            .ok_or_else(|| Error::new(ruby.exception_runtime_error(), "Dataset not opened"))?;

        let count = self.runtime.borrow_mut().block_on(async {
            dataset.count_rows(None)
                .await
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))
        })?;

        Ok(count as i64)
    }

    pub fn schema(&self) -> Result<RHash, Error> {
        let ruby = Ruby::get().unwrap();
        let dataset = self.dataset.borrow();
        let dataset = dataset.as_ref()
            .ok_or_else(|| Error::new(ruby.exception_runtime_error(), "Dataset not opened"))?;

        // Get the actual schema from the Lance dataset
        let schema = self.runtime.borrow_mut().block_on(async {
            dataset.schema()
        });

        // Convert Lance schema to Arrow schema
        let arrow_schema: arrow_schema::Schema = schema.into();
        let arrow_schema = Arc::new(arrow_schema);

        let hash = ruby.hash_new();

        // Iterate over Arrow schema fields
        for field in arrow_schema.fields() {
            let field_name = ruby.to_symbol(field.name());

            // Handle vector columns specially
            if let DataType::FixedSizeList(inner_field, dimension) = field.data_type() {
                // Check if it's a vector (float list)
                if matches!(inner_field.data_type(), DataType::Float32 | DataType::Float16) {
                    let vector_info = ruby.hash_new();
                    vector_info.aset(ruby.to_symbol("type"), "vector")?;
                    vector_info.aset(ruby.to_symbol("dimension"), *dimension)?;
                    hash.aset(field_name, vector_info)?;
                    continue;
                }
            }

            let field_type = datatype_to_ruby_string(field.data_type());
            hash.aset(field_name, field_type)?;
        }

        Ok(hash)
    }

    pub fn scan_all(&self) -> Result<RArray, Error> {
        let ruby = Ruby::get().unwrap();
        let dataset = self.dataset.borrow();
        let dataset = dataset.as_ref()
            .ok_or_else(|| Error::new(ruby.exception_runtime_error(), "Dataset not opened"))?;

        let batches: Vec<RecordBatch> = self.runtime.borrow_mut().block_on(async {
            let scanner = dataset.scan();
            let stream = scanner
                .try_into_stream()
                .await
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;

            stream
                .try_collect::<Vec<_>>()
                .await
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))
        })?;

        let result_array = ruby.ary_new();

        for batch in batches {
            let batch_docs = convert_batch_to_ruby(&ruby, &batch)?;
            // Merge arrays by pushing each element
            for i in 0..batch_docs.len() {
                result_array.push(batch_docs.entry::<Value>(i as isize)?)?;
            }
        }

        Ok(result_array)
    }

    pub fn scan_limit(&self, limit: i64) -> Result<RArray, Error> {
        let ruby = Ruby::get().unwrap();
        let dataset = self.dataset.borrow();
        let dataset = dataset.as_ref()
            .ok_or_else(|| Error::new(ruby.exception_runtime_error(), "Dataset not opened"))?;

        let batches: Vec<RecordBatch> = self.runtime.borrow_mut().block_on(async {
            let mut scanner = dataset.scan();
            scanner.limit(Some(limit), None)
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;

            let stream = scanner
                .try_into_stream()
                .await
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;

            stream
                .try_collect::<Vec<_>>()
                .await
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))
        })?;

        let result_array = ruby.ary_new();

        for batch in batches {
            let batch_docs = convert_batch_to_ruby(&ruby, &batch)?;
            // Merge arrays by pushing each element
            for i in 0..batch_docs.len() {
                result_array.push(batch_docs.entry::<Value>(i as isize)?)?;
            }
        }

        Ok(result_array)
    }

    pub fn create_vector_index(&self, column: String) -> Result<(), Error> {
        let ruby = Ruby::get().unwrap();
        let mut dataset = self.dataset.borrow_mut();
        let dataset = dataset.as_mut()
            .ok_or_else(|| Error::new(ruby.exception_runtime_error(), "Dataset not opened"))?;

        self.runtime.borrow_mut().block_on(async move {
            // Get row count to determine optimal number of partitions
            let num_rows = dataset.count_rows(None).await
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;

            // Use fewer partitions for small datasets
            let num_partitions = if num_rows < 256 {
                std::cmp::max(1, (num_rows / 4) as usize)
            } else {
                256
            };

            // Create IVF_FLAT vector index parameters
            let params = VectorIndexParams::ivf_flat(num_partitions, lance_linalg::distance::MetricType::L2);

            dataset.create_index(
                &[&column],
                IndexType::Vector,
                None,
                &params,
                true
            )
            .await
            .map(|_| ())
            .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))
        })
    }

    pub fn vector_search(&self, column: String, query_vector: RArray, limit: i64) -> Result<RArray, Error> {
        let ruby = Ruby::get().unwrap();
        let dataset = self.dataset.borrow();
        let dataset = dataset.as_ref()
            .ok_or_else(|| Error::new(ruby.exception_runtime_error(), "Dataset not opened"))?;

        // Convert Ruby array to Vec<f32> using index-based iteration
        let len = query_vector.len();
        let mut vector: Vec<f32> = Vec::with_capacity(len);
        for i in 0..len {
            let v: f64 = query_vector.entry(i as isize)?;
            vector.push(v as f32);
        }

        let batches: Vec<RecordBatch> = self.runtime.borrow_mut().block_on(async {
            let mut scanner = dataset.scan();

            // Use nearest for vector search
            scanner.nearest(&column, &Float32Array::from(vector), limit as usize)
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;

            let stream = scanner
                .try_into_stream()
                .await
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;

            stream
                .try_collect::<Vec<_>>()
                .await
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))
        })?;

        let result_array = ruby.ary_new();

        for batch in batches {
            let batch_docs = convert_batch_to_ruby(&ruby, &batch)?;
            // Merge arrays by pushing each element
            for i in 0..batch_docs.len() {
                result_array.push(batch_docs.entry::<Value>(i as isize)?)?;
            }
        }

        Ok(result_array)
    }

    pub fn create_text_index(&self, column: String) -> Result<(), Error> {
        let ruby = Ruby::get().unwrap();
        let mut dataset = self.dataset.borrow_mut();
        let dataset = dataset.as_mut()
            .ok_or_else(|| Error::new(ruby.exception_runtime_error(), "Dataset not opened"))?;

        self.runtime.borrow_mut().block_on(async move {
            // Create inverted index for full-text search
            let params = InvertedIndexParams::default();

            dataset.create_index(
                &[&column],
                IndexType::Inverted,
                None,
                &params,
                true
            )
            .await
            .map(|_| ())
            .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))
        })
    }

    pub fn text_search(&self, column: String, query: String, limit: i64) -> Result<RArray, Error> {
        let ruby = Ruby::get().unwrap();
        let dataset = self.dataset.borrow();
        let dataset = dataset.as_ref()
            .ok_or_else(|| Error::new(ruby.exception_runtime_error(), "Dataset not opened"))?;

        let batches: Vec<RecordBatch> = self.runtime.borrow_mut().block_on(async {
            let mut scanner = dataset.scan();

            // Use full-text search with inverted index
            let fts_query = FullTextSearchQuery::new(query)
                .with_column(column)
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;

            scanner.full_text_search(fts_query)
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;

            // Apply limit
            scanner.limit(Some(limit), None)
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;

            let stream = scanner
                .try_into_stream()
                .await
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;

            stream
                .try_collect::<Vec<_>>()
                .await
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))
        })?;

        let result_array = ruby.ary_new();

        for batch in batches {
            let batch_docs = convert_batch_to_ruby(&ruby, &batch)?;
            // Merge arrays by pushing each element
            for i in 0..batch_docs.len() {
                result_array.push(batch_docs.entry::<Value>(i as isize)?)?;
            }
        }

        Ok(result_array)
    }

    pub fn multi_column_text_search(&self, columns: RArray, query: String, limit: i64) -> Result<RArray, Error> {
        let ruby = Ruby::get().unwrap();
        let dataset = self.dataset.borrow();
        let dataset = dataset.as_ref()
            .ok_or_else(|| Error::new(ruby.exception_runtime_error(), "Dataset not opened"))?;

        // Convert Ruby array of columns to Vec<String> using index-based iteration
        let len = columns.len();
        let mut cols: Vec<String> = Vec::with_capacity(len);
        for i in 0..len {
            let v: String = columns.entry(i as isize)?;
            cols.push(v);
        }

        let batches: Vec<RecordBatch> = self.runtime.borrow_mut().block_on(async {
            let mut scanner = dataset.scan();

            // Create a full-text search query for multiple columns
            let fts_query = FullTextSearchQuery::new(query)
                .with_columns(&cols)
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;

            scanner.full_text_search(fts_query)
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;

            // Apply limit
            scanner.limit(Some(limit), None)
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;

            let stream = scanner
                .try_into_stream()
                .await
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;

            stream
                .try_collect::<Vec<_>>()
                .await
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))
        })?;

        let result_array = ruby.ary_new();

        for batch in batches {
            let batch_docs = convert_batch_to_ruby(&ruby, &batch)?;
            // Merge arrays by pushing each element
            for i in 0..batch_docs.len() {
                result_array.push(batch_docs.entry::<Value>(i as isize)?)?;
            }
        }

        Ok(result_array)
    }

    pub fn filter_scan(&self, filter_expr: String, limit: Option<i64>) -> Result<RArray, Error> {
        let ruby = Ruby::get().unwrap();
        let dataset = self.dataset.borrow();
        let dataset = dataset.as_ref()
            .ok_or_else(|| Error::new(ruby.exception_runtime_error(), "Dataset not opened"))?;

        let batches: Vec<RecordBatch> = self.runtime.borrow_mut().block_on(async {
            let mut scanner = dataset.scan();

            // Apply SQL-like filter
            scanner.filter(&filter_expr)
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;

            // Apply limit if provided
            if let Some(lim) = limit {
                scanner.limit(Some(lim), None)
                    .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;
            }

            let stream = scanner
                .try_into_stream()
                .await
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;

            stream
                .try_collect::<Vec<_>>()
                .await
                .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))
        })?;

        let result_array = ruby.ary_new();

        for batch in batches {
            let batch_docs = convert_batch_to_ruby(&ruby, &batch)?;
            // Merge arrays by pushing each element
            for i in 0..batch_docs.len() {
                result_array.push(batch_docs.entry::<Value>(i as isize)?)?;
            }
        }

        Ok(result_array)
    }
}

impl LancelotDataset {
    pub fn bind(class: &RClass) -> Result<(), Error> {
        class.define_singleton_method("new", function!(LancelotDataset::new, 1))?;
        class.define_method("path", method!(LancelotDataset::path, 0))?;
        class.define_method("create", method!(LancelotDataset::create, 1))?;
        class.define_method("open", method!(LancelotDataset::open, 0))?;
        class.define_method("add_data", method!(LancelotDataset::add_data, 1))?;
        class.define_method("count_rows", method!(LancelotDataset::count_rows, 0))?;
        class.define_method("schema", method!(LancelotDataset::schema, 0))?;
        class.define_method("scan_all", method!(LancelotDataset::scan_all, 0))?;
        class.define_method("scan_limit", method!(LancelotDataset::scan_limit, 1))?;
        class.define_method("create_vector_index", method!(LancelotDataset::create_vector_index, 1))?;
        class.define_method("create_text_index", method!(LancelotDataset::create_text_index, 1))?;
        class.define_method("_rust_vector_search", method!(LancelotDataset::vector_search, 3))?;
        class.define_method("_rust_text_search", method!(LancelotDataset::text_search, 3))?;
        class.define_method("_rust_multi_column_text_search", method!(LancelotDataset::multi_column_text_search, 3))?;
        class.define_method("filter_scan", method!(LancelotDataset::filter_scan, 2))?;
        Ok(())
    }
}
