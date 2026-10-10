"""
pdf_processor.py
Standalone module for AI-powered PDF processing and intelligent linking
No web framework dependencies - pure Python module
"""

import os
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import json
import re
from typing import List, Dict, Tuple, Optional


class PDFTextExtractor:
    """Extract and clean text from PDF files"""
    
    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """
        Extract all text content from a PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                print(f"✓ Extracted text from {os.path.basename(pdf_path)} ({total_pages} pages, {len(text)} chars)")
                return text.strip()
                
        except Exception as e:
            print(f"✗ Error extracting from {pdf_path}: {str(e)}")
            return ""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', '', text)
        # Remove multiple punctuation
        text = re.sub(r'([.,!?;:])\1+', r'\1', text)
        return text.strip()
    
    @staticmethod
    def get_metadata(pdf_path: str) -> Dict:
        """Extract PDF metadata"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata
                
                return {
                    'title': metadata.get('/Title', '') if metadata else '',
                    'author': metadata.get('/Author', '') if metadata else '',
                    'pages': len(pdf_reader.pages),
                    'filename': os.path.basename(pdf_path)
                }
        except:
            return {
                'title': '',
                'author': '',
                'pages': 0,
                'filename': os.path.basename(pdf_path)
            }


class AIDocumentAnalyzer:
    """AI-powered document analysis and linking using embeddings"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize AI analyzer with transformer model
        
        Args:
            model_name: Sentence transformer model name
                - 'all-MiniLM-L6-v2': Fast, good quality (recommended)
                - 'all-mpnet-base-v2': Best quality, slower
                - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual
        """
        print(f"Loading AI model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("✓ AI model loaded\n")
    
    def generate_document_embedding(self, text: str) -> np.ndarray:
        """Generate semantic embedding for document"""
        return self.model.encode(text, show_progress_bar=False)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        similarity = cosine_similarity(
            embedding1.reshape(1, -1), 
            embedding2.reshape(1, -1)
        )[0][0]
        return float(similarity)
    
    def extract_topics(self, texts: List[str], n_topics: int = 3) -> Tuple[Optional[np.ndarray], Optional[List[List[str]]]]:
        """
        Extract topics from documents using Latent Dirichlet Allocation
        
        Args:
            texts: List of document texts
            n_topics: Number of topics to extract
            
        Returns:
            Tuple of (topic_distributions, top_words_per_topic)
        """
        if len(texts) < 2:
            return None, None
            
        try:
            vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words='english',
                max_df=0.85,
                min_df=1,
                ngram_range=(1, 2)
            )
            
            doc_term_matrix = vectorizer.fit_transform(texts)
            
            n_topics = min(n_topics, len(texts))
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=50
            )
            
            topic_distributions = lda.transform(doc_term_matrix)
            
            # Extract top words for each topic
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_weights in lda.components_:
                top_indices = topic_weights.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                topics.append(top_words)
            
            return topic_distributions, topics
            
        except Exception as e:
            print(f"Warning: Topic extraction failed - {e}")
            return None, None
    
    def cluster_documents(self, embeddings: np.ndarray, n_clusters: int = 3) -> np.ndarray:
        """Cluster documents based on embeddings"""
        n_clusters = min(n_clusters, len(embeddings))
        if n_clusters < 2:
            return np.zeros(len(embeddings))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(embeddings)


class PDFMindMapGenerator:
    """Main class to process PDFs and generate mind map data"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize PDF Mind Map Generator
        
        Args:
            model_name: AI model to use for analysis
        """
        self.extractor = PDFTextExtractor()
        self.analyzer = AIDocumentAnalyzer(model_name)
    
    def process_pdf_files(self, pdf_paths: List[str]) -> List[Dict]:
        """
        Process multiple PDF files and extract content
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of document dictionaries with extracted data
        """
        documents = []
        
        print(f"Processing {len(pdf_paths)} PDF files...\n")
        
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"✗ File not found: {pdf_path}")
                continue
            
            # Extract text
            text = self.extractor.extract_text(pdf_path)
            
            if not text or len(text) < 100:
                print(f"⚠ Warning: {os.path.basename(pdf_path)} has insufficient text")
                continue
            
            # Clean text
            cleaned_text = self.extractor.clean_text(text)
            
            # Get metadata
            metadata = self.extractor.get_metadata(pdf_path)
            
            # Create document object
            doc = {
                'id': os.path.basename(pdf_path),
                'filename': metadata['filename'],
                'title': metadata['title'] or metadata['filename'].replace('.pdf', ''),
                'text': cleaned_text,
                'word_count': len(cleaned_text.split()),
                'page_count': metadata['pages'],
                'path': pdf_path
            }
            
            documents.append(doc)
        
        print(f"\n✓ Successfully processed {len(documents)} documents\n")
        return documents
    
    def generate_mind_map(
        self, 
        pdf_paths: List[str], 
        similarity_threshold: float = 0.25,
        n_topics: int = 3,
        n_clusters: int = 3
    ) -> Dict:
        """
        Generate complete mind map data from PDF files
        
        Args:
            pdf_paths: List of PDF file paths
            similarity_threshold: Minimum similarity to create link (0-1)
            n_topics: Number of topics to extract
            n_clusters: Number of document clusters
            
        Returns:
            Dictionary with nodes, links, clusters, and metadata
        """
        # Process PDFs
        documents = self.process_pdf_files(pdf_paths)
        
        if len(documents) < 2:
            return {
                'error': 'Need at least 2 valid PDF files',
                'nodes': [],
                'links': [],
                'clusters': {}
            }
        
        print("Running AI analysis...\n")
        
        # Generate embeddings
        print("Generating semantic embeddings...")
        texts = [doc['text'] for doc in documents]
        embeddings = np.array([self.analyzer.generate_document_embedding(text) for text in texts])
        print(f"   ✓ Created {len(embeddings)} embeddings\n")
        
        # Extract topics
        print("Extracting topics...")
        topic_distributions, topics = self.analyzer.extract_topics(texts, n_topics)
        if topics:
            print(f"   ✓ Extracted {len(topics)} topics\n")
        else:
            print("   ⚠ Topic extraction skipped\n")
        
        # Cluster documents
        print("Clustering documents...")
        cluster_labels = self.analyzer.cluster_documents(embeddings, n_clusters)
        print(f"   ✓ Created {len(set(cluster_labels))} clusters\n")
        
        # Create links
        print("Analyzing relationships...")
        links = []
        
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                # Calculate semantic similarity
                semantic_sim = self.analyzer.calculate_similarity(embeddings[i], embeddings[j])
                
                # Calculate topic similarity if available
                topic_sim = 0.0
                shared_topics = []
                
                if topic_distributions is not None:
                    topic_sim = self.analyzer.calculate_similarity(
                        topic_distributions[i], 
                        topic_distributions[j]
                    )
                    
                    # Find shared dominant topics
                    top_topics_i = set(np.argsort(topic_distributions[i])[-2:])
                    top_topics_j = set(np.argsort(topic_distributions[j])[-2:])
                    shared_topic_indices = top_topics_i & top_topics_j
                    
                    if shared_topic_indices and topics:
                        for topic_idx in shared_topic_indices:
                            shared_topics.extend(topics[topic_idx][:3])
                
                # Combined AI score
                ai_score = (semantic_sim * 0.7) + (topic_sim * 0.3)
                
                # Create link if above threshold
                if ai_score >= similarity_threshold:
                    link = {
                        'source': documents[i]['id'],
                        'target': documents[j]['id'],
                        'semantic_similarity': round(semantic_sim, 3),
                        'topic_similarity': round(topic_sim, 3),
                        'ai_score': round(ai_score, 3),
                        'shared_topics': list(set(shared_topics))[:5],
                        'link_type': self._classify_link_strength(ai_score),
                        'confidence': self._calculate_confidence(semantic_sim, topic_sim)
                    }
                    links.append(link)
        
        # Sort links by AI score
        links.sort(key=lambda x: x['ai_score'], reverse=True)
        print(f"   ✓ Created {len(links)} connections\n")
        
        # Prepare nodes
        nodes = []
        for i, doc in enumerate(documents):
            nodes.append({
                'id': doc['id'],
                'label': doc['title'][:50],  # Truncate long titles
                'filename': doc['filename'],
                'word_count': doc['word_count'],
                'page_count': doc['page_count'],
                'cluster': int(cluster_labels[i]),
                'summary': doc['text'][:200] + '...'  # First 200 chars
            })
        
        # Organize clusters
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'id': documents[i]['id'],
                'filename': documents[i]['filename']
            })
        
        # Prepare final output
        result = {
            'nodes': nodes,
            'links': links,
            'clusters': {str(k): v for k, v in clusters.items()},
            'topics': topics if topics else [],
            'statistics': {
                'total_documents': len(documents),
                'total_links': len(links),
                'total_clusters': len(set(cluster_labels)),
                'avg_similarity': round(np.mean([l['ai_score'] for l in links]), 3) if links else 0,
                'max_similarity': round(max([l['ai_score'] for l in links]), 3) if links else 0
            }
        }
        
        print("Mind map generation complete!\n")
        return result
    
    def _classify_link_strength(self, score: float) -> str:
        """Classify link strength based on AI score"""
        if score >= 0.7:
            return "strong"
        elif score >= 0.5:
            return "moderate"
        elif score >= 0.3:
            return "weak"
        else:
            return "minimal"
    
    def _calculate_confidence(self, semantic_sim: float, topic_sim: float) -> str:
        """Calculate confidence based on agreement between metrics"""
        diff = abs(semantic_sim - topic_sim)
        if diff < 0.2:
            return "high"
        elif diff < 0.4:
            return "medium"
        else:
            return "low"
    
    def save_to_json(self, data: Dict, output_path: str = 'mind_map_data.json'):
        """Save mind map data to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved to: {output_path}")


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    # Initialize generator
    generator = PDFMindMapGenerator(model_name='all-MiniLM-L6-v2')
    
    # List of PDF files to process (UPDATE THESE PATHS)
    pdf_files = [
        'document1.pdf',
        'document2.pdf',
        'document3.pdf',
        'document4.pdf'
    ]
    
    # Check if files exist
    existing_files = [f for f in pdf_files if os.path.exists(f)]
    
    if len(existing_files) < 2:
        print("Error: Please provide at least 2 valid PDF files")
        print("Update the pdf_files list with actual PDF paths")
        exit(1)
    
    # Generate mind map
    mind_map_data = generator.generate_mind_map(
        pdf_paths=existing_files,
        similarity_threshold=0.25,  # Adjust based on your needs (0.2-0.5 recommended)
        n_topics=3,
        n_clusters=3
    )
    
    # Check for errors
    if 'error' in mind_map_data:
        print(f"  Error: {mind_map_data['error']}")
        exit(1)
    
    # Save to JSON
    generator.save_to_json(mind_map_data, 'ai_mind_map.json')
    
    # Display results
    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(json.dumps(mind_map_data['statistics'], indent=2))
    
    print("\n" + "=" * 70)
    print("SAMPLE LINKS")
    print("=" * 70)
    for link in mind_map_data['links'][:5]:  # Show top 5 links
        print(f"\n{link['source']} ←→ {link['target']}")
        print(f"  AI Score: {link['ai_score']} ({link['link_type']})")
        print(f"  Confidence: {link['confidence']}")
        if link['shared_topics']:
            print(f"  Topics: {', '.join(link['shared_topics'][:3])}")
    
    print("\n" + "=" * 70)
    print("CLUSTERS")
    print("=" * 70)
    for cluster_id, docs in mind_map_data['clusters'].items():
        print(f"\nCluster {cluster_id}:")
        for doc in docs:
            print(f"{doc['filename']}")