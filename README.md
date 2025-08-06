# Term Base Creator

A comprehensive application for extracting, managing, and analyzing terminology from text corpora. Built with Python and tkinter, this tool helps linguists, translators, and content managers create and maintain professional termbases.

<img width="1057" height="764" alt="fileprocessing_screen" src="https://github.com/user-attachments/assets/6e113e22-66f0-4a8a-89c7-b315c5e1b41d" />

## Features

### Core Functionality
- **Multi-format Corpus Loading**: Support for TXT, DOCX, XLIFF, and HTML files
- **Automated Term Extraction**: Uses spaCy NLP for intelligent term identification
- **Manual Term Search**: Find specific terms with context-aware search
- **Term Validation**: Review and validate extracted terms with rich editing capabilities
- **Co-occurrence Analysis**: Analyze term relationships and collocations
- **Import/Export**: Work with existing termbases and export results

### Advanced Features
- **Smart Term Recognition**: Identifies named entities, noun phrases, and honorific combinations
- **Context Management**: Multiple contexts per term with highlighting
- **Bulk Operations**: Edit multiple terms simultaneously
- **PMI Analysis**: Pointwise Mutual Information scoring for term associations
- **Flexible Filtering**: View terms by validation status, source, and frequency

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Required Dependencies

Install all required packages:

```bash
pip install spacy pandas python-docx beautifulsoup4 lxml
```

### spaCy Language Model

Download the required English language model (recommended):

```bash
python -m spacy download en_core_web_md
```

Or the smaller alternative:

```bash
python -m spacy download en_core_web_sm
```

## Quick Start

1. **Run the application:**
   ```bash
   python termbase_creator.py
   ```
   
<img width="386" height="289" alt="image" src="https://github.com/user-attachments/assets/0590a206-c97a-44fe-b776-39238b7ebf5e" />

2. **Load your corpus:**
   - Click "Load Corpus" in the top toolbar
   - Select your text file (TXT, DOCX, XLIFF, or HTML)
   - Review the corpus statistics

3. **Extract terms:**
   - Go to the "Auto-Extract" tab
   - Set minimum frequency (default: 2)
   - Click "Extract Terms"

4. **Review and validate:**
   - Switch to "Term Validation" tab
   - Review extracted terms
   - Add target translations and notes
   - Mark terms as validated

5. **Export your termbase:**
   - Click "Export Selected" or "Export Current"
   - Choose CSV or Excel format

## User Interface Guide

### Tabs Overview

#### 1. Auto-Extract Tab
- **Purpose**: Automated term extraction from loaded corpus
- **Features**:
  - Frequency-based filtering
  - Debug mode for extraction details
  - Context preview with highlighting
  - Sortable term list
  - Checkbox selection for export

#### 2. Manual Search Tab
- **Purpose**: Search for specific terms in the corpus
- **Features**:
  - Case-sensitive and whole-word search options
  - Context extraction with position information
  - Add search results directly to termbase
  - Preview of search contexts
<img width="1400" height="931" alt="manualsearch_screen" src="https://github.com/user-attachments/assets/3cf3ad37-7c06-4e23-b64c-bda04a019aa5" />

#### 3. Term Validation Tab
- **Purpose**: Review, edit, and validate terms
- **Features**:
  - Filter by validation status (All, Validated, Pending, New, Imported)
  - Quick edit fields for target terms and notes
  - Bulk edit operations
  - Export validated terms

#### 4. Co-occurrence Tab
- **Purpose**: Analyze term relationships
- **Features**:
  - Adjacent bigram analysis
  - Sentence-level co-occurrence
  - PMI (Pointwise Mutual Information) scoring
  - Exportable co-occurrence data

## File Format Support

### Input Formats
- **TXT**: Plain text files
- **DOCX**: Microsoft Word documents
- **XLIFF**: Translation memory exchange format
- **HTML/HTM**: Web pages and HTML documents

### Output Formats
- **CSV**: Comma-separated values (UTF-8 encoded)
- **XLSX**: Microsoft Excel format

## Term Extraction Methodology

The application uses a multi-method approach:

1. **Honorific + Name Patterns**: Regex-based extraction of titles with names
2. **Named Entity Recognition**: spaCy's NER for persons, organizations, locations
3. **Noun Phrase Extraction**: Clean and filter noun chunks
4. **Co-occurrence Analysis**: Track adjacent and sentence-level term relationships

### Quality Filters
- Minimum frequency thresholds
- Stopword removal (with honorific exceptions)
- Redundancy elimination
- Length and quality validation

## Advanced Usage

### Bulk Operations

Select multiple terms in the validation tab and use bulk edit to:
- Mark all as validated/pending
- Update source information
- Append notes to multiple terms
- Find and replace text patterns (with regex support)

### Co-occurrence Analysis

1. Extract terms first using Auto-Extract
2. Switch to Co-occurrence tab
3. Choose analysis type:
   - **Adjacent Bigrams**: Terms appearing next to each other
   - **Sentence Co-occurrence**: Terms in the same sentence
4. Set minimum co-occurrence threshold
5. Review PMI scores for semantic strength

### Import Existing Termbases

1. Click "Import Termbase" in the top toolbar
2. Select CSV file with columns:
   - `Source Term` (required)
   - `Target Term`
   - `Frequency`
   - `Primary Context`
   - `Notes`
   - `Validated`
3. Choose merge strategy:
   - **Merge**: Combine with existing terms
   - **Replace**: Replace all current terms
   - **Separate**: Keep as separate termbase

## CSV Export Structure

Exported CSV files contain:
- `Source Term`: Original extracted term
- `Target Term`: Translation or equivalent
- `Frequency`: Occurrence count in corpus
- `Primary Context`: Main context sentence
- `Total Contexts`: Number of available contexts
- `Source`: Extraction method (auto-extracted, manual-search, imported)
- `Validated`: Boolean validation status
- `Notes`: User-added notes and comments

## Performance Considerations

### Large Corpora
- Text is processed in chunks (500,000 characters each)
- Progress bars show processing status
- Memory-efficient processing for files up to several GB

### Processing Times (Approximate)
- Less than 1,000 words: < 1 second
- 1,000-10,000 words: 1-5 seconds
- 10,000-50,000 words: 5-30 seconds
- 50,000-100,000 words: 30-60 seconds
- More than 100,000 words: 1+ minutes

## Troubleshooting

### Common Issues

**spaCy Model Not Found**
```
python -m spacy download en_core_web_md
```

**Memory Issues with Large Files**
- The application chunks large files automatically
- Close other applications to free memory
- Consider using the smaller spaCy model (en_core_web_sm)

**Extraction Taking Too Long**
- Enable debug mode to see progress
- Increase minimum frequency to reduce results
- Process smaller text sections

**Unicode/Encoding Issues**
- Files are processed with UTF-8 encoding
- Exported CSV files use UTF-8 encoding
- Check source file encoding if characters appear corrupted

### Debug Mode

Enable debug mode in the Auto-Extract tab to see:
- Chunk processing details
- Honorific pattern matches
- Named entity extraction results
- Processing timing information

## Technical Details

### Dependencies
- **spaCy**: Natural language processing and term extraction
- **pandas**: Data manipulation and CSV/Excel export
- **python-docx**: Microsoft Word document processing
- **beautifulsoup4**: HTML parsing and text extraction
- **tkinter**: GUI framework (included with Python)

### Architecture
- **Main Application**: `TermExtractorApp` class managing the GUI
- **Extraction Engine**: Multiple extraction methods with spaCy integration
- **Data Management**: In-memory term storage with export capabilities
- **Dialog Classes**: Specialized interfaces for editing and importing

## Contributing

### Code Structure
- Utility functions for text processing and term extraction
- Dialog classes for specialized UI components
- Main application class with tabbed interface
- Modular design for easy extension

### Potential Enhancements
- Additional file format support (PDF, RTF)
- Term frequency visualization
- Database storage backend
- Multi-language support

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

### You are free to:
- **Use** the software for non-commercial purposes
- **Share** and distribute the software
- **Adapt** and modify the software

### Under the following terms:
- **Attribution** — You must give appropriate credit
- **NonCommercial** — You may not use the material for commercial purposes

For commercial licensing inquiries, please contact [luisa.strindade@gmail.com]

### What counts as non-commercial?
✅ Individual translators, linguists, and researchers  
✅ Academic institutions and students  
✅ Personal projects and learning  
❌ Companies using it for business operations  
❌ Selling products or services that include this software  

See the [full license](https://creativecommons.org/licenses/by-nc/4.0/) for details.

## Support

For issues or questions about the Term Base Creator:
1. Check the troubleshooting section above
2. Enable debug mode for detailed processing information
3. Verify all dependencies are correctly installed
4. Ensure spaCy language models are available

---

**Note**: This application requires the spaCy medium model (en_core_web_md) for optimal performance. The small model (en_core_web_sm) can be used as a fallback but may produce lower-quality term extractions.
