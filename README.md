# GalHero: Photo Gallery Search with CLIP and Mac Photos Integration

A Python-based photo search system that combines CLIP's powerful image understanding capabilities with Mac Photos metadata. Search your photo library using natural language queries and metadata filters.

## Features

- Natural language image search using OpenAI's CLIP model
- Mac Photos metadata integration (locations, people, dates, albums)
- Support for HEIC/HEIF image formats
- Advanced filtering options (date, location, people, favorites)
- Automatic metadata import from Mac Photos
- Efficient caching of embeddings for fast repeated searches

## Installation

1. Install required packages:
   ```bash
   pip install transformers torch Pillow osxphotos pillow-heif
   ```
## Grant Necessary Permissions

1. Open System Preferences > Security & Privacy > Privacy.
2. Under "Photos", ensure Terminal/Python has access.

## Usage

1. Update the `GALLERY_DIR` in `main()` to point to your photos directory:
   ```python
   GALLERY_DIR = "path/to/your/photos"  # Replace with your photos directory
    ```
2. Run the python script
    ```python
    python all.py
    ```

### Choose from Available Options

- **Simple search**: Search using natural language.
- **Search with filters**: Add metadata filters to your search.
- **Regenerate embeddings**: Update embeddings for new photos.
- **Exit**: Close the application.


### Using Filters

You can filter results by:
- Date range.
- People tagged in photos.
- Locations.
- Favorites.

## How It Works

1. **CLIP Model**: Uses OpenAI's CLIP model to understand image content and text queries.
2. **Metadata Integration**: Imports metadata from Mac Photos for enhanced search capabilities.
3. **Embedding Generation**: Creates and caches embeddings for all images.
4. **Search Process**: Combines CLIP's semantic understanding with metadata filters.

## Technical Details

- **CLIP Model**: Utilizes OpenAI's vision-language model for semantic search.
- **Image Formats**: Supports various formats including JPG, PNG, and HEIC.
- **Embedding Normalization**: Normalizes embeddings for accurate similarity matching.
- **Caching**: Caches results to avoid regenerating embeddings for repeated searches.
- **Metadata Handling**: Processes EXIF data and integrates Mac Photos metadata.

## Limitations

- **Setup Time**: Initial setup may be time-consuming for large photo libraries.
- **Permissions**: Requires access permissions for Mac Photos.
- **Memory Usage**: Scales with the size of the photo library.

## Future Enhancements

- **Face Clustering**: Group untagged photos by detected faces.
- **Content-Based Similarity**: Enable search by similar content.
- **Scene Detection**: Auto-tagging based on detected scenes.
- **Time-Based Organization**: Organize photos by temporal proximity.
- **Memory Features**: Create memory-like photo collections.
- **Export Options**: Add capabilities for exporting and sharing.
- **Smart Albums**: Automatically generate albums based on content.
- **Location Clustering**: Group photos by geographic locations.

## Contributing

Feel free to fork the repository, submit pull requests, or suggest new features. Contributions are always welcome!

## License

This project is licensed under the MIT License.

## Acknowledgments

- **OpenAI**: For the CLIP model.
- **Hugging Face**: For hosting and providing access to models.
- **osxphotos**: For seamless integration with Mac Photos.