# frozen_string_literal: true

require "tempfile"
require "fileutils"

RSpec.describe Lancelot::Dataset do
  let(:temp_dir) { Dir.mktmpdir }
  let(:dataset_path) { File.join(temp_dir, "test_dataset") }

  after do
    GC.start
    FileUtils.rm_rf(temp_dir)
  end

  describe ".create" do
    it "creates a new dataset with a schema" do
      schema = {
        text: :string,
        score: :float32,
        embedding: { type: "vector", dimension: 3 }
      }

      dataset = Lancelot::Dataset.create(dataset_path, schema: schema)
      expect(dataset).to be_a(Lancelot::Dataset)
      expect(dataset.path).to eq(dataset_path)
    end

    it "normalizes schema types" do
      schema = {
        text: "string",
        score: :float
      }

      dataset = Lancelot::Dataset.create(dataset_path, schema: schema)
      expect(dataset.schema).to eq({
        text: "string",
        score: "float32"
      })
    end
  end

  describe ".open" do
    it "opens an existing dataset" do
      schema = { text: :string, score: :float32 }
      Lancelot::Dataset.create(dataset_path, schema: schema)

      dataset = Lancelot::Dataset.open(dataset_path)
      expect(dataset).to be_a(Lancelot::Dataset)
      expect(dataset.schema).to eq({ text: "string", score: "float32" })
    end
  end

  describe ".open_or_create" do
    let(:schema) { { text: :string, embedding: { type: "vector", dimension: 3 } } }

    it "creates a new dataset if it doesn't exist" do
      expect(File.exist?(dataset_path)).to be false
      
      dataset = Lancelot::Dataset.open_or_create(dataset_path, schema: schema)
      expect(dataset).to be_a(Lancelot::Dataset)
      expect(File.exist?(dataset_path)).to be true
    end

    it "opens existing dataset if it already exists" do
      # First create a dataset with some data
      initial_dataset = Lancelot::Dataset.create(dataset_path, schema: schema)
      initial_dataset.add_documents([{ text: "test", embedding: [1.0, 2.0, 3.0] }])
      expect(initial_dataset.count).to eq(1)
      
      # Now open_or_create should open it, not recreate it
      dataset = Lancelot::Dataset.open_or_create(dataset_path, schema: schema)
      expect(dataset).to be_a(Lancelot::Dataset)
      expect(dataset.count).to eq(1)  # Data should still be there
    end

    it "is idempotent - can be called multiple times safely" do
      # First call creates
      dataset1 = Lancelot::Dataset.open_or_create(dataset_path, schema: schema)
      dataset1.add_documents([{ text: "doc1", embedding: [0.1, 0.2, 0.3] }])
      
      # Second call opens
      dataset2 = Lancelot::Dataset.open_or_create(dataset_path, schema: schema)
      expect(dataset2.count).to eq(1)
      
      # Third call also opens
      dataset3 = Lancelot::Dataset.open_or_create(dataset_path, schema: schema)
      expect(dataset3.count).to eq(1)
    end

    context "with existing non-empty directory" do
      before do
        FileUtils.mkdir_p(dataset_path)
        File.write(File.join(dataset_path, "existing_file.txt"), "important data")
      end

      it "raises an error when directory contains non-dataset files" do
        expect {
          Lancelot::Dataset.open_or_create(dataset_path, schema: schema)
        }.to raise_error(ArgumentError, /Directory exists.*but is not a valid Lance dataset/)
      end

      it "suggests using mode: 'overwrite' in error message" do
        expect {
          Lancelot::Dataset.open_or_create(dataset_path, schema: schema)
        }.to raise_error(ArgumentError, /Use mode: 'overwrite' to replace it/)
      end

      it "overwrites directory when mode: 'overwrite' is specified" do
        existing_file = File.join(dataset_path, "existing_file.txt")
        expect(File.exist?(existing_file)).to be true

        dataset = Lancelot::Dataset.open_or_create(dataset_path, schema: schema, mode: "overwrite")
        
        expect(dataset).to be_a(Lancelot::Dataset)
        expect(File.exist?(existing_file)).to be false
        expect(File.exist?(File.join(dataset_path, "_versions"))).to be true
      end
    end

    context "with empty directory" do
      before do
        FileUtils.mkdir_p(dataset_path)
      end

      it "creates dataset in empty directory without error" do
        expect(Dir.empty?(dataset_path)).to be true
        
        dataset = Lancelot::Dataset.open_or_create(dataset_path, schema: schema)
        
        expect(dataset).to be_a(Lancelot::Dataset)
        expect(File.exist?(File.join(dataset_path, "_versions"))).to be true
      end
    end

    context "with existing valid dataset" do
      before do
        initial_dataset = Lancelot::Dataset.create(dataset_path, schema: schema)
        initial_dataset.add_documents([
          { text: "doc1", embedding: [0.1, 0.2, 0.3] },
          { text: "doc2", embedding: [0.4, 0.5, 0.6] }
        ])
      end

      it "opens existing dataset without overwriting" do
        dataset = Lancelot::Dataset.open_or_create(dataset_path, schema: schema)
        
        expect(dataset.count).to eq(2)
        docs = dataset.all
        expect(docs.map { |d| d[:text] }).to contain_exactly("doc1", "doc2")
      end

      it "ignores mode: 'overwrite' when opening valid dataset" do
        # Even with overwrite mode, should open existing valid dataset
        dataset = Lancelot::Dataset.open_or_create(dataset_path, schema: schema, mode: "overwrite")
        
        expect(dataset.count).to eq(2)
      end
    end

    context "with non-existent path" do
      it "creates new dataset at non-existent path" do
        expect(File.exist?(dataset_path)).to be false
        
        dataset = Lancelot::Dataset.open_or_create(dataset_path, schema: schema)
        
        expect(dataset).to be_a(Lancelot::Dataset)
        expect(File.exist?(dataset_path)).to be true
        expect(File.exist?(File.join(dataset_path, "_versions"))).to be true
      end

      it "creates parent directories if needed" do
        nested_path = File.join(dataset_path, "nested", "deep", "dataset")
        expect(File.exist?(File.dirname(nested_path))).to be false
        
        dataset = Lancelot::Dataset.open_or_create(nested_path, schema: schema)
        
        expect(dataset).to be_a(Lancelot::Dataset)
        expect(File.exist?(nested_path)).to be true
      end
    end

    context "with file instead of directory" do
      before do
        FileUtils.mkdir_p(File.dirname(dataset_path))
        File.write(dataset_path, "I'm a file, not a directory")
      end

      it "raises an error when path is a file" do
        expect {
          Lancelot::Dataset.open_or_create(dataset_path, schema: schema)
        }.to raise_error(ArgumentError)
      end

      it "can overwrite file with mode: 'overwrite'" do
        expect(File.file?(dataset_path)).to be true
        
        dataset = Lancelot::Dataset.open_or_create(dataset_path, schema: schema, mode: "overwrite")
        
        expect(dataset).to be_a(Lancelot::Dataset)
        expect(File.directory?(dataset_path)).to be true
        expect(File.exist?(File.join(dataset_path, "_versions"))).to be true
      end
    end
  end

  describe "#add_documents" do
    let(:dataset) do
      schema = {
        text: :string,
        score: :float32,
        embedding: { type: "vector", dimension: 3 }
      }
      Lancelot::Dataset.create(dataset_path, schema: schema)
    end

    it "adds documents to the dataset" do
      documents = [
        { text: "Hello world", score: 0.9, embedding: [0.1, 0.2, 0.3] },
        { text: "Ruby is great", score: 0.8, embedding: [0.4, 0.5, 0.6] }
      ]

      dataset.add_documents(documents)
      expect(dataset.count).to eq(2)
    end

    it "accepts string keys" do
      documents = [
        { "text" => "Hello", "score" => 0.5, "embedding" => [1.0, 2.0, 3.0] }
      ]

      expect { dataset.add_documents(documents) }.not_to raise_error
      expect(dataset.count).to eq(1)
    end

    context "with optional fields (schema evolution)" do
      it "allows adding documents with missing fields that were added later" do
        # This test verifies the fix for optional fields in conversion.rs
        # Previously, this would fail with "key not found" errors
        
        # Step 1: Create dataset with N fields (text, score, embedding)
        initial_docs = [
          { text: "Document 1", score: 0.9, embedding: [0.1, 0.2, 0.3] },
          { text: "Document 2", score: 0.8, embedding: [0.4, 0.5, 0.6] }
        ]
        dataset.add_documents(initial_docs)
        expect(dataset.count).to eq(2)
        
        # Step 2: Close and reopen dataset
        dataset = Lancelot::Dataset.open(dataset_path)
        
        # Step 3: Add documents with N+1 fields (adding 'metadata' field)
        # In a real scenario, this would happen after schema evolution
        # For now, we simulate by updating all existing docs with new field
        all_docs = dataset.to_a
        updated_docs = all_docs.map { |doc| doc.merge(metadata: "extra") }
        
        # Recreate dataset with expanded schema
        FileUtils.rm_rf(dataset_path)
        expanded_schema = {
          text: :string,
          score: :float32,
          embedding: { type: "vector", dimension: 3 },
          metadata: :string  # New field
        }
        dataset = Lancelot::Dataset.create(dataset_path, schema: expanded_schema)
        dataset.add_documents(updated_docs)
        expect(dataset.count).to eq(2)
        
        # Step 4: Add new documents with only N fields (missing 'metadata')
        # This is the key test - documents without the new field should work
        new_docs = [
          { text: "Document 3", score: 0.7, embedding: [0.7, 0.8, 0.9] },
          { text: "Document 4", score: 0.6, embedding: [1.0, 1.1, 1.2] }
        ]
        
        # This should NOT raise an error with the fix
        expect { dataset.add_documents(new_docs) }.not_to raise_error
        expect(dataset.count).to eq(4)
        
        # Verify the documents were added correctly
        all_data = dataset.to_a
        expect(all_data.size).to eq(4)
        
        # Documents 1-2 should have metadata
        expect(all_data[0][:metadata]).to eq("extra")
        expect(all_data[1][:metadata]).to eq("extra")
        
        # Documents 3-4 should have nil metadata (optional field)
        expect(all_data[2][:metadata]).to be_nil
        expect(all_data[3][:metadata]).to be_nil
      end

      it "handles documents with varying fields in a single batch" do
        # Create dataset with a schema that has optional fields
        schema = {
          id: :string,
          text: :string,
          score: :float32,
          category: :string  # This will be optional
        }
        dataset = Lancelot::Dataset.create(dataset_path, schema: schema)
        
        # Add documents where some have all fields and some are missing fields
        mixed_docs = [
          { id: "1", text: "Full document", score: 0.9, category: "complete" },
          { id: "2", text: "Partial document", score: 0.8 },  # Missing category
          { id: "3", text: "Another full", score: 0.7, category: "complete" },
          { id: "4", text: "Another partial", score: 0.6 }  # Missing category
        ]
        
        expect { dataset.add_documents(mixed_docs) }.not_to raise_error
        expect(dataset.count).to eq(4)
        
        # Verify the data
        all_data = dataset.to_a
        expect(all_data[0][:category]).to eq("complete")
        expect(all_data[1][:category]).to be_nil
        expect(all_data[2][:category]).to eq("complete")
        expect(all_data[3][:category]).to be_nil
      end
    end
  end

  describe "#<<" do
    let(:dataset) do
      schema = { text: :string, score: :float32 }
      Lancelot::Dataset.create(dataset_path, schema: schema)
    end

    it "adds a single document" do
      dataset << { text: "Hello", score: 0.7 }
      expect(dataset.count).to eq(1)
    end

    it "returns self for chaining" do
      result = dataset << { text: "First", score: 0.1 } << { text: "Second", score: 0.2 }
      expect(result).to eq(dataset)
      expect(dataset.count).to eq(2)
    end
  end

  describe "#size, #count, #length" do
    let(:dataset) do
      schema = { text: :string, score: :float32 }
      Lancelot::Dataset.create(dataset_path, schema: schema)
    end

    it "returns the number of rows" do
      expect(dataset.size).to eq(0)
      
      dataset.add_documents([
        { text: "Hello", score: 0.5 }, 
        { text: "World", score: 0.8 }
      ])
      
      expect(dataset.size).to eq(2)
      expect(dataset.count).to eq(2)
      expect(dataset.length).to eq(2)
    end
  end

  describe "#schema" do
    it "returns the actual dataset schema for basic types" do
      schema = {
        title: :string,
        count: :int32,
        rating: :float64,
        score: :float32,
        active: :boolean
      }
      
      dataset = Lancelot::Dataset.create(dataset_path, schema: schema)
      returned_schema = dataset.schema
      
      
      expect(returned_schema[:title]).to eq("string")
      expect(returned_schema[:count]).to eq("int32") 
      expect(returned_schema[:rating]).to eq("float64")
      expect(returned_schema[:score]).to eq("float32")
      expect(returned_schema[:active]).to eq("boolean")
    end
    
    it "returns correct schema for vector columns" do
      schema = {
        text: :string,
        embedding: { type: "vector", dimension: 768 },
        small_vector: { type: "vector", dimension: 3 }
      }
      
      dataset = Lancelot::Dataset.create(dataset_path, schema: schema)
      returned_schema = dataset.schema
      
      expect(returned_schema[:text]).to eq("string")
      expect(returned_schema[:embedding]).to be_a(Hash)
      expect(returned_schema[:embedding][:type]).to eq("vector")
      expect(returned_schema[:embedding][:dimension]).to eq(768)
      expect(returned_schema[:small_vector][:dimension]).to eq(3)
    end
    
    it "returns correct schema after reopening dataset" do
      schema = {
        id: :string,
        value: :float32,
        embedding: { type: "vector", dimension: 128 }
      }
      
      # Create dataset with schema
      dataset1 = Lancelot::Dataset.create(dataset_path, schema: schema)
      dataset1.add_documents([
        { id: "test1", value: 1.0, embedding: Array.new(128, 0.1) }
      ])
      original_schema = dataset1.schema
      
      # Reopen and verify schema matches
      dataset2 = Lancelot::Dataset.open(dataset_path)
      reopened_schema = dataset2.schema
      
      expect(reopened_schema).to eq(original_schema)
      expect(reopened_schema[:id]).to eq("string")
      expect(reopened_schema[:value]).to eq("float32")
      expect(reopened_schema[:embedding][:type]).to eq("vector")
      expect(reopened_schema[:embedding][:dimension]).to eq(128)
    end
    
    it "preserves schema after adding data" do
      schema = {
        text: :string,
        score: :float32
      }
      
      dataset = Lancelot::Dataset.create(dataset_path, schema: schema)
      initial_schema = dataset.schema
      
      # Add various documents
      dataset.add_documents([
        { text: "doc1", score: 0.5 },
        { text: "doc2", score: 0.7 }
      ])
      
      dataset << { text: "doc3", score: 0.9 }
      
      final_schema = dataset.schema
      
      # Schema should remain unchanged after data operations
      expect(final_schema).to eq(initial_schema)
    end
    
    it "handles all supported types correctly" do
      schema = {
        string_field: :string,
        int32_field: :int32,
        int64_field: :int64, 
        float32_field: :float32,
        float64_field: :float64,
        bool_field: :boolean,
        vector_field: { type: "vector", dimension: 10 }
      }
      
      dataset = Lancelot::Dataset.create(dataset_path, schema: schema)
      returned_schema = dataset.schema
      
      expect(returned_schema[:string_field]).to eq("string")
      expect(returned_schema[:int32_field]).to eq("int32")
      expect(returned_schema[:int64_field]).to eq("int64")
      expect(returned_schema[:float32_field]).to eq("float32")
      expect(returned_schema[:float64_field]).to eq("float64")
      expect(returned_schema[:bool_field]).to eq("boolean")
      expect(returned_schema[:vector_field]).to be_a(Hash)
      expect(returned_schema[:vector_field][:type]).to eq("vector")
      expect(returned_schema[:vector_field][:dimension]).to eq(10)
    end
    
    it "returns consistent key format (symbols)" do
      schema = {
        my_field: :string,
        another_field: :float32
      }
      
      dataset = Lancelot::Dataset.create(dataset_path, schema: schema)
      returned_schema = dataset.schema
      
      # Keys should be symbols, not strings
      expect(returned_schema.keys).to all(be_a(Symbol))
    end
    
    context "with open_or_create" do
      it "returns correct schema when creating new dataset" do
        schema = {
          name: :string,
          value: :float32
        }
        
        dataset = Lancelot::Dataset.open_or_create(dataset_path, schema: schema)
        returned_schema = dataset.schema
        
        expect(returned_schema[:name]).to eq("string")
        expect(returned_schema[:value]).to eq("float32")
      end
      
      it "returns correct schema when opening existing dataset" do
        schema = {
          name: :string,
          value: :float32,
          vector: { type: "vector", dimension: 5 }
        }
        
        # First create
        dataset1 = Lancelot::Dataset.create(dataset_path, schema: schema)
        dataset1 << { name: "test", value: 1.0, vector: [0.1, 0.2, 0.3, 0.4, 0.5] }
        
        # Then open with open_or_create
        dataset2 = Lancelot::Dataset.open_or_create(dataset_path, schema: schema)
        
        expect(dataset2.schema).to eq(dataset1.schema)
        expect(dataset2.schema[:vector][:dimension]).to eq(5)
      end
    end
  end

  describe "document retrieval" do
    let(:dataset) do
      schema = { text: :string, score: :float32 }
      Lancelot::Dataset.create(dataset_path, schema: schema)
    end

    before do
      dataset.add_documents([
        { text: "Ruby is great", score: 0.95 },
        { text: "Python is cool", score: 0.75 },
        { text: "JavaScript is everywhere", score: 0.85 }
      ])
    end

    describe "#all" do
      it "returns all documents" do
        docs = dataset.all
        expect(docs).to be_an(Array)
        expect(docs.length).to eq(3)
        expect(docs.first[:text]).to eq("Ruby is great")
      end
    end

    describe "#first" do
      it "returns the first document when called without argument" do
        doc = dataset.first
        expect(doc).to be_a(Hash)
        expect(doc[:text]).to eq("Ruby is great")
      end

      it "returns the first n documents when called with argument" do
        docs = dataset.first(2)
        expect(docs).to be_an(Array)
        expect(docs.length).to eq(2)
        expect(docs[0][:text]).to eq("Ruby is great")
        expect(docs[1][:text]).to eq("Python is cool")
      end
    end

    describe "#each" do
      it "yields each document" do
        texts = []
        dataset.each { |doc| texts << doc[:text] }
        expect(texts).to eq(["Ruby is great", "Python is cool", "JavaScript is everywhere"])
      end

      it "returns an enumerator when no block given" do
        enum = dataset.each
        expect(enum).to be_an(Enumerator)
        expect(enum.to_a.length).to eq(3)
      end
    end

    describe "Enumerable methods" do
      it "supports map" do
        texts = dataset.map { |doc| doc[:text] }
        expect(texts).to eq(["Ruby is great", "Python is cool", "JavaScript is everywhere"])
      end

      it "supports select" do
        high_score_docs = dataset.select { |doc| doc[:score] && doc[:score] >= 0.9 }
        expect(high_score_docs.length).to eq(1)
        expect(high_score_docs.first[:text]).to eq("Ruby is great")
      end
    end

    describe "streaming behavior" do
      context "with moderate-sized dataset" do
        before do
          # Add enough documents to likely span multiple batches
          # Lance typically uses batches of 1024 rows, but we'll use 100 for testing
          100.times do |i|
            dataset << { text: "Document #{i}", score: i.to_f / 100 }
          end
        end

        it "supports early termination with take" do
          # Should only process first 5 documents, not all 103
          taken = dataset.take(5)
          expect(taken.length).to eq(5)
          expect(taken.first[:text]).to eq("Ruby is great")
        end

        it "supports early termination with first(n)" do
          # Should efficiently get first 10 without processing all
          first_ten = dataset.first(10)
          expect(first_ten.length).to eq(10)
        end

        it "stops iteration on break" do
          count = 0
          dataset.each do |doc|
            count += 1
            break if count == 7
          end
          expect(count).to eq(7)
        end

        it "works with lazy enumerator chains" do
          # Lazy evaluation should prevent loading all documents
          result = dataset.lazy.select { |doc| doc[:score] && doc[:score] > 0.5 }.first(3)
          expect(result.length).to eq(3)
          expect(result.all? { |doc| doc[:score] > 0.5 }).to be true
        end

        it "handles each_slice efficiently" do
          slices = []
          dataset.each_slice(10) do |slice|
            slices << slice
            break if slices.length == 2  # Only process first 20 documents
          end
          expect(slices.length).to eq(2)
          expect(slices.first.length).to eq(10)
        end
      end

      context "with empty dataset" do
        let(:empty_dataset) do
          Lancelot::Dataset.create(
            File.join(temp_dir, "empty_dataset"),
            schema: { text: :string, score: :float32 }
          )
        end

        after { FileUtils.rm_rf(File.join(temp_dir, "empty_dataset")) }

        it "handles each on empty dataset" do
          count = 0
          empty_dataset.each { count += 1 }
          expect(count).to eq(0)
        end

        it "returns empty array for to_a" do
          expect(empty_dataset.to_a).to eq([])
        end

        it "returns nil for first on empty dataset" do
          expect(empty_dataset.first).to be_nil
        end

        it "returns empty array for first(n) on empty dataset" do
          expect(empty_dataset.first(5)).to eq([])
        end
      end

      context "with single document" do
        let(:single_dataset) do
          ds = Lancelot::Dataset.create(
            File.join(temp_dir, "single_dataset"),
            schema: { text: :string, score: :float32 }
          )
          ds << { text: "Only one", score: 1.0 }
          ds
        end

        after { FileUtils.rm_rf(File.join(temp_dir, "single_dataset")) }

        it "iterates over single document" do
          docs = single_dataset.to_a
          expect(docs.length).to eq(1)
          expect(docs.first[:text]).to eq("Only one")
        end

        it "handles first correctly" do
          expect(single_dataset.first[:text]).to eq("Only one")
        end

        it "handles first(n) correctly" do
          expect(single_dataset.first(5).length).to eq(1)
        end
      end

      context "enumerable method behavior" do
        let(:enum_dataset) do
          ds = Lancelot::Dataset.create(
            File.join(temp_dir, "enum_dataset"),
            schema: { text: :string, score: :float32 }
          )
          ds.add_documents([
            { text: "Ruby is great", score: 0.95 },
            { text: "Python is cool", score: 0.7 },
            { text: "JavaScript is everywhere", score: 0.8 }
          ])
          ds
        end

        after { FileUtils.rm_rf(File.join(temp_dir, "enum_dataset")) }

        it "supports find" do
          found = enum_dataset.find { |doc| doc[:text].include?("Python") }
          expect(found[:text]).to eq("Python is cool")
        end

        it "supports find with no match" do
          found = enum_dataset.find { |doc| doc[:text].include?("Rust") }
          expect(found).to be_nil
        end

        it "supports any?" do
          expect(enum_dataset.any? { |doc| doc[:score] && doc[:score] > 0.8 }).to be true
          expect(enum_dataset.any? { |doc| doc[:score] && doc[:score] > 1.0 }).to be false
        end

        it "supports all?" do
          expect(enum_dataset.all? { |doc| doc[:text].is_a?(String) }).to be true
          expect(enum_dataset.all? { |doc| doc[:score] && doc[:score] > 0.5 }).to be true
          expect(enum_dataset.all? { |doc| doc[:score] && doc[:score] > 0.9 }).to be false
        end

        it "supports none?" do
          expect(enum_dataset.none? { |doc| doc[:score] && doc[:score] > 1.0 }).to be true
          expect(enum_dataset.none? { |doc| doc[:text].include?("Ruby") }).to be false
        end

        it "supports count with block" do
          count = enum_dataset.count { |doc| doc[:score] && doc[:score] >= 0.9 }
          expect(count).to eq(1)  # Only Ruby (0.95) is >= 0.9
        end

        it "allows multiple concurrent enumerators" do
          enum1 = enum_dataset.each
          enum2 = enum_dataset.each
          
          # Both should work independently
          doc1 = enum1.next
          doc2 = enum2.next
          expect(doc1).to eq(doc2)  # Both start from beginning
          
          # Advance enum1
          enum1.next
          doc1_third = enum1.next
          
          # enum2 should still be on second position
          doc2_second = enum2.next
          expect(doc2_second).not_to eq(doc1_third)
        end
      end

      context "with each.with_index" do
        it "provides correct indices" do
          indices = []
          texts = []
          dataset.each.with_index do |doc, i|
            indices << i
            texts << doc[:text]
            break if i >= 2
          end
          expect(indices).to eq([0, 1, 2])
          expect(texts).to eq(["Ruby is great", "Python is cool", "JavaScript is everywhere"])
        end
      end
    end
  end

  describe "vector search" do
    let(:dataset) do
      schema = { 
        text: :string, 
        score: :float32,
        vector: { type: "vector", dimension: 3 }
      }
      Lancelot::Dataset.create(dataset_path, schema: schema)
    end

    before do
      dataset.add_documents([
        { text: "Ruby programming", score: 0.9, vector: [0.1, 0.2, 0.3] },
        { text: "Python coding", score: 0.85, vector: [0.2, 0.3, 0.4] },
        { text: "JavaScript development", score: 0.8, vector: [0.8, 0.9, 0.7] }
      ])
    end

    describe "#create_vector_index" do
      it "creates a vector index" do
        expect { dataset.create_vector_index("vector") }.not_to raise_error
      end
    end

    describe "#vector_search" do
      before do
        dataset.create_vector_index("vector")
      end

      it "finds nearest neighbors" do
        query_vector = [0.15, 0.25, 0.35]
        results = dataset.vector_search(query_vector, column: "vector", limit: 2)
        
        expect(results).to be_an(Array)
        expect(results.length).to eq(2)
        # The first two documents should be the closest
        texts = results.map { |doc| doc[:text] }
        expect(texts).to include("Ruby programming", "Python coding")
      end

      it "respects the limit parameter" do
        query_vector = [0.5, 0.5, 0.5]
        results = dataset.vector_search(query_vector, column: "vector", limit: 1)
        
        expect(results.length).to eq(1)
      end

      it "raises error for non-array query" do
        expect {
          dataset.vector_search("not an array", column: "vector")
        }.to raise_error(ArgumentError, /must be an array/)
      end
    end

    describe "#nearest_neighbors" do
      before do
        dataset.create_vector_index("vector")
      end

      it "calls vector_search with k parameter" do
        query_vector = [0.1, 0.2, 0.3]
        results = dataset.nearest_neighbors(query_vector, k: 2, column: "vector")
        
        expect(results).to be_an(Array)
        expect(results.length).to eq(2)
      end
    end
  end

  describe "text search" do
    let(:dataset) do
      schema = { 
        title: :string,
        content: :string,
        category: :string,
        year: :int64
      }
      Lancelot::Dataset.create(dataset_path, schema: schema)
    end

    before do
      dataset.add_documents([
        { title: "Ruby on Rails", content: "Web framework for Ruby", category: "web", year: 2023 },
        { title: "Django Python", content: "Web framework for Python", category: "web", year: 2024 },
        { title: "Ruby Gems", content: "Package manager for Ruby", category: "tools", year: 2023 },
        { title: "Python Packages", content: "PyPI is the Python package index", category: "tools", year: 2024 }
      ])
    end

    describe "#create_text_index" do
      it "creates a text index on a column" do
        expect { dataset.create_text_index("title") }.not_to raise_error
        expect { dataset.create_text_index("content") }.not_to raise_error
      end
    end

    describe "#text_search" do
      context "with text indices" do
        before do
          dataset.create_text_index("title")
          dataset.create_text_index("content")
          dataset.create_text_index("category")
        end

        it "searches a single column" do
          results = dataset.text_search("ruby", column: "title")
          expect(results).to be_an(Array)
          expect(results.length).to eq(2)
          titles = results.map { |doc| doc[:title] }
          expect(titles).to include("Ruby on Rails", "Ruby Gems")
        end

        it "searches with default column" do
          # Default is "text" column which doesn't exist
          # This will raise an error because the column doesn't exist
          expect {
            dataset.text_search("ruby")
          }.to raise_error(RuntimeError, /Column text not found/)
        end

        it "searches multiple columns" do
          results = dataset.text_search("framework", columns: ["title", "content"])
          expect(results.length).to be >= 2
        end

        it "is case insensitive" do
          results = dataset.text_search("RUBY", column: "title")
          expect(results.length).to eq(2)
        end

        it "handles multi-word queries" do
          results = dataset.text_search("package manager", column: "content")
          expect(results.length).to be >= 1
          expect(results.first[:title]).to eq("Ruby Gems")
        end

        it "raises error for non-string query" do
          expect {
            dataset.text_search(123, column: "title")
          }.to raise_error(ArgumentError, /must be a string/)
        end

        it "raises error when both column and columns specified" do
          expect {
            dataset.text_search("ruby", column: "title", columns: ["content"])
          }.to raise_error(ArgumentError, /Cannot specify both/)
        end
      end
    end

    describe "#where" do
      it "filters with simple conditions" do
        results = dataset.where("year = 2023")
        expect(results.length).to eq(2)
        expect(results.map { |doc| doc[:title] }).to include("Ruby on Rails", "Ruby Gems")
      end

      it "filters with compound conditions" do
        results = dataset.where("category = 'web' AND year = 2024")
        expect(results.length).to eq(1)
        expect(results.first[:title]).to eq("Django Python")
      end

      it "filters with LIKE patterns" do
        results = dataset.where("title LIKE '%Python%'")
        expect(results.length).to eq(2)
      end

      it "supports limit parameter" do
        results = dataset.where("category = 'web'", limit: 1)
        expect(results.length).to eq(1)
      end

      it "handles OR conditions" do
        results = dataset.where("title LIKE '%Rails%' OR title LIKE '%Django%'")
        expect(results.length).to eq(2)
      end
    end
  end

  describe "Ruby object methods" do
    let(:dataset) do
      schema = { text: :string, score: :float32 }
      ds = Lancelot::Dataset.create(dataset_path, schema: schema)
      # Add at least one document so the dataset is properly initialized
      ds.add_documents([{ text: "test", score: 0.5 }])
      ds
    end

    describe "#to_s and #inspect" do
      it "returns a string representation with path and count" do
        dataset.add_documents([
          { text: "First", score: 0.5 },
          { text: "Second", score: 0.8 }
        ])
        
        expected = "#<Lancelot::Dataset path=\"#{dataset_path}\" count=3>"
        expect(dataset.to_s).to eq(expected)
        expect(dataset.inspect).to eq(expected)
      end

      it "shows count for dataset with initial document" do
        expected = "#<Lancelot::Dataset path=\"#{dataset_path}\" count=1>"
        expect(dataset.to_s).to eq(expected)
      end
    end

    describe "#== and #eql?" do
      it "returns true for datasets with the same path" do
        # Ensure dataset is fully written by calling count
        expect(dataset.count).to eq(1)
        
        dataset2 = Lancelot::Dataset.open(dataset_path)
        
        expect(dataset == dataset2).to be true
        expect(dataset.eql?(dataset2)).to be true
      end

      it "returns false for datasets with different paths" do
        other_path = File.join(temp_dir, "other_dataset")
        other_dataset = Lancelot::Dataset.create(other_path, schema: { text: :string })
        
        expect(dataset == other_dataset).to be false
        expect(dataset.eql?(other_dataset)).to be false
      end

      it "returns false when comparing with non-dataset objects" do
        expect(dataset == "not a dataset").to be false
        expect(dataset == nil).to be false
        expect(dataset == 123).to be false
      end
    end

    describe "#hash" do
      it "returns the same hash for datasets with the same path" do
        # Ensure dataset is fully written
        expect(dataset.count).to eq(1)
        
        dataset2 = Lancelot::Dataset.open(dataset_path)
        
        expect(dataset.hash).to eq(dataset2.hash)
      end

      it "returns different hashes for datasets with different paths" do
        other_path = File.join(temp_dir, "other_dataset")
        other_dataset = Lancelot::Dataset.create(other_path, schema: { text: :string })
        
        expect(dataset.hash).not_to eq(other_dataset.hash)
      end

      it "can be used as a hash key" do
        # Ensure dataset is fully written
        expect(dataset.count).to eq(1)
        
        dataset2 = Lancelot::Dataset.open(dataset_path)
        
        hash = {}
        hash[dataset] = "value1"
        hash[dataset2] = "value2"
        
        # Should overwrite since they're the same dataset
        expect(hash.size).to eq(1)
        expect(hash[dataset]).to eq("value2")
        expect(hash[dataset2]).to eq("value2")
      end

      it "can be used in a Set" do
        require 'set'
        
        # Ensure dataset is fully written
        expect(dataset.count).to eq(1)
        
        dataset2 = Lancelot::Dataset.open(dataset_path)
        
        set = Set.new
        set.add(dataset)
        set.add(dataset2)
        
        # Should only have one element since they're the same dataset
        expect(set.size).to eq(1)
      end
    end

    describe "#path" do
      it "returns the dataset path" do
        expect(dataset.path).to eq(dataset_path)
      end
    end
  end

  describe "#hybrid_search" do
    let(:dataset) do
      schema = {
        title: :string,
        content: :string,
        embedding: { type: "vector", dimension: 3 }
      }
      Lancelot::Dataset.create(dataset_path, schema: schema)
    end

    before do
      documents = [
        { 
          title: "Ruby on Rails", 
          content: "A web framework for Ruby",
          embedding: [0.1, 0.2, 0.3]
        },
        { 
          title: "Python Django", 
          content: "A web framework for Python",
          embedding: [0.4, 0.5, 0.6]
        },
        { 
          title: "Ruby Gems", 
          content: "Package manager for Ruby",
          embedding: [0.2, 0.3, 0.4]
        },
        { 
          title: "JavaScript Express", 
          content: "A minimal web framework",
          embedding: [0.7, 0.8, 0.9]
        }
      ]
      
      dataset.add_documents(documents)
      dataset.create_vector_index("embedding")
      dataset.create_text_index("title")
      dataset.create_text_index("content")
    end

    it "combines vector and text search results" do
      query_vector = [0.15, 0.25, 0.35]
      results = dataset.hybrid_search(
        "Ruby",
        vector: query_vector,
        vector_column: "embedding",
        text_column: "title",
        limit: 3
      )
      
      expect(results).to be_an(Array)
      expect(results.length).to be <= 3
      
      # Results should have RRF scores
      results.each do |doc|
        expect(doc).to have_key(:rrf_score)
        expect(doc[:rrf_score]).to be_a(Float)
        expect(doc[:rrf_score]).to be > 0
      end
      
      # Should be sorted by RRF score descending
      scores = results.map { |doc| doc[:rrf_score] }
      expect(scores).to eq(scores.sort.reverse)
    end

    it "works with only vector search" do
      query_vector = [0.1, 0.2, 0.3]
      results = dataset.hybrid_search(
        nil,
        vector: query_vector,
        vector_column: "embedding",
        limit: 2
      )
      
      expect(results).to be_an(Array)
      expect(results.length).to be <= 2
      # Should not have RRF scores when only one search type
      expect(results.first).not_to have_key(:rrf_score)
    end

    it "works with only text search" do
      results = dataset.hybrid_search(
        "framework",
        text_column: "content",
        limit: 2
      )
      
      expect(results).to be_an(Array)
      expect(results.length).to be <= 2
      # Should not have RRF scores when only one search type
      expect(results.first).not_to have_key(:rrf_score)
    end

    it "supports multi-column text search" do
      query_vector = [0.2, 0.3, 0.4]
      results = dataset.hybrid_search(
        "Ruby",
        vector: query_vector,
        vector_column: "embedding",
        text_columns: ["title", "content"],
        limit: 3
      )
      
      expect(results).to be_an(Array)
      expect(results.length).to be <= 3
      results.each do |doc|
        expect(doc).to have_key(:rrf_score)
      end
    end

    it "returns empty array when no results match" do
      results = dataset.hybrid_search(
        "NonexistentTerm",
        text_column: "title",
        limit: 10
      )
      
      expect(results).to eq([])
    end

    it "returns empty array when neither query nor vector provided" do
      results = dataset.hybrid_search(nil, limit: 10)
      expect(results).to eq([])
      
      results = dataset.hybrid_search("", limit: 10)
      expect(results).to eq([])
    end

    it "respects custom RRF k parameter" do
      query_vector = [0.1, 0.2, 0.3]
      results = dataset.hybrid_search(
        "Ruby",
        vector: query_vector,
        vector_column: "embedding",
        text_column: "title",
        limit: 2,
        rrf_k: 100
      )
      
      expect(results).to be_an(Array)
      results.each do |doc|
        expect(doc).to have_key(:rrf_score)
      end
    end

    it "raises error for invalid vector" do
      expect {
        dataset.hybrid_search("Ruby", vector: "not an array", text_column: "title")
      }.to raise_error(ArgumentError, /Vector must be an array/)
    end

    it "handles documents appearing in only one result set" do
      # Use a query that will return different documents in each search
      query_vector = [0.9, 0.9, 0.9]  # Closer to JavaScript document
      results = dataset.hybrid_search(
        "Ruby",  # Will match Ruby documents
        vector: query_vector,
        vector_column: "embedding",
        text_column: "title",
        limit: 4
      )
      
      expect(results).to be_an(Array)
      # Should include documents from both searches
      titles = results.map { |doc| doc[:title] }
      expect(titles).to include("JavaScript Express") # From vector search
      expect(titles).to include("Ruby on Rails")      # From text search
      expect(titles).to include("Ruby Gems")          # From text search
    end

    it "properly deduplicates documents across result sets" do
      # Use similar vector to Ruby documents
      query_vector = [0.15, 0.25, 0.35]
      results = dataset.hybrid_search(
        "Ruby",
        vector: query_vector,
        vector_column: "embedding",
        text_column: "title",
        limit: 10
      )
      
      # Count occurrences of each document
      title_counts = results.group_by { |doc| doc[:title] }.transform_values(&:count)
      
      # Each document should appear only once
      title_counts.each do |_title, count|
        expect(count).to eq(1)
      end
    end
  end
end