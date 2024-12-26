import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import numpy as np
from pathlib import Path
import pickle
import osxphotos
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from pillow_heif import register_heif_opener

# Register HEIF/HEIC opener with Pillow
register_heif_opener()

@dataclass
class PhotoMetadata:
    path: str
    timestamp: Optional[datetime] = None
    location: Optional[str] = None
    people: List[str] = field(default_factory=list)
    description: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    favorite: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    album_names: List[str] = field(default_factory=list)

class ImageGallery:
    def __init__(self, image_dir: str):
        self.image_dir = Path(image_dir)
        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.embeddings = {}
        self.metadata = {}
        
    def import_photos_metadata(self):
        """Import metadata from Mac Photos app"""
        print("Importing metadata from Photos...")
        photosdb = osxphotos.PhotosDB()
        
        for photo in photosdb.photos():
            if photo.path:
                try:
                    metadata = PhotoMetadata(
                        path=str(photo.path),
                        timestamp=photo.date,
                        location=photo.place.name if photo.place else None,
                        people=list(photo.persons) if photo.persons else [],
                        description=photo.description,
                        keywords=list(photo.keywords) if photo.keywords else [],
                        favorite=photo.favorite,
                        latitude=photo.location.latitude if photo.location else None,
                        longitude=photo.location.longitude if photo.location else None,
                        album_names=[album.title for album in photo.albums]
                    )
                    self.metadata[str(photo.path)] = metadata
                    print(f"Imported metadata for: {photo.path}")
                except Exception as e:
                    print(f"Error processing photo metadata {photo.path}: {e}")

    def process_image(self, image_path: Path):
        """Process a single image, handling different formats"""
        try:
            # Open image (supports HEIC)
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate embedding
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            embeddings = image_features.numpy()
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return embeddings
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def generate_embeddings(self):
        """Generate embeddings for all images"""
        print("Generating embeddings...")
        image_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.HEIC'}
        
        for image_path in self.image_dir.rglob('*'):
            if image_path.suffix.lower() in image_extensions:
                print(f"Processing: {image_path}")
                embeddings = self.process_image(image_path)
                
                if embeddings is not None:
                    self.embeddings[str(image_path)] = embeddings
                    
                    # Create basic metadata if not already exists
                    if str(image_path) not in self.metadata:
                        self.metadata[str(image_path)] = PhotoMetadata(
                            path=str(image_path),
                            timestamp=datetime.fromtimestamp(image_path.stat().st_mtime)
                        )
    
    def search(self, query: str, top_k: int = 5, filter_params: dict = None):
        """Search images by text query with optional filters"""
        print(f"Searching for: {query}")
        
        inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        text_embeddings = text_features.numpy()
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        
        results = []
        for path, embedding in self.embeddings.items():
            if filter_params and not self._apply_filters(path, filter_params):
                continue
                
            similarity = np.dot(text_embeddings, embedding.T)[0][0]
            metadata = self.metadata[path]
            results.append((path, similarity, metadata))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _apply_filters(self, image_path: str, filter_params: dict) -> bool:
        """Apply metadata filters"""
        metadata = self.metadata[image_path]
        
        for key, value in filter_params.items():
            if key == 'date_range':
                start_date, end_date = value
                if not (metadata.timestamp and start_date <= metadata.timestamp <= end_date):
                    return False
            elif key == 'people':
                if not all(person in metadata.people for person in value):
                    return False
            elif key == 'location':
                if metadata.location != value:
                    return False
            elif key == 'favorite':
                if metadata.favorite != value:
                    return False
            elif key == 'albums':
                if not any(album in metadata.album_names for album in value):
                    return False
        return True
    
    def save(self, save_path: str):
        """Save gallery data"""
        print(f"Saving gallery to {save_path}")
        data = {
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, load_path: str, image_dir: str):
        """Load gallery data"""
        print(f"Loading gallery from {load_path}")
        gallery = cls(image_dir)
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            gallery.embeddings = data['embeddings']
            gallery.metadata = data['metadata']
        return gallery

def main():
    # Configure these paths
    GALLERY_DIR = "/Users/thevedantsingh/Desktop/album"  # Replace with your photos directory
    SAVE_PATH = "gallery_data.pkl"
    
    # Load or create gallery
    if os.path.exists(SAVE_PATH):
        gallery = ImageGallery.load(SAVE_PATH, GALLERY_DIR)
    else:
        gallery = ImageGallery(GALLERY_DIR)
        gallery.import_photos_metadata()
        gallery.generate_embeddings()
        gallery.save(SAVE_PATH)
    
    while True:
        print("\nImage Search Options:")
        print("1. Simple search")
        print("2. Search with filters")
        print("3. Regenerate embeddings")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            query = input("Enter your search query: ")
            results = gallery.search(query)
            
            print("\nSearch Results:")
            for path, score, metadata in results:
                print(f"\nImage: {path}")
                print(f"Score: {score:.3f}")
                if metadata.location:
                    print(f"Location: {metadata.location}")
                if metadata.people:
                    print(f"People: {', '.join(metadata.people)}")
                if metadata.timestamp:
                    print(f"Date: {metadata.timestamp.strftime('%Y-%m-%d')}")
        
        elif choice == '2':
            query = input("Enter your search query: ")
            
            # Collect filters
            filters = {}
            
            # Date range filter
            use_dates = input("Filter by date range? (y/n): ").lower() == 'y'
            if use_dates:
                start = input("Start date (YYYY-MM-DD): ")
                end = input("End date (YYYY-MM-DD): ")
                filters['date_range'] = (
                    datetime.strptime(start, '%Y-%m-%d'),
                    datetime.strptime(end, '%Y-%m-%d')
                )
            
            # People filter
            people = input("Filter by people (comma-separated, or press enter to skip): ")
            if people:
                filters['people'] = [p.strip() for p in people.split(',')]
            
            # Location filter
            location = input("Filter by location (press enter to skip): ")
            if location:
                filters['location'] = location
            
            # Favorites filter
            if input("Show only favorites? (y/n): ").lower() == 'y':
                filters['favorite'] = True
            
            results = gallery.search(query, filter_params=filters)
            
            print("\nSearch Results:")
            for path, score, metadata in results:
                print(f"\nImage: {path}")
                print(f"Score: {score:.3f}")
                if metadata.location:
                    print(f"Location: {metadata.location}")
                if metadata.people:
                    print(f"People: {', '.join(metadata.people)}")
                if metadata.timestamp:
                    print(f"Date: {metadata.timestamp.strftime('%Y-%m-%d')}")
        
        elif choice == '3':
            gallery.generate_embeddings()
            gallery.save(SAVE_PATH)
            print("Embeddings regenerated and saved.")
            
        elif choice == '4':
            break

if __name__ == "__main__":
    main()