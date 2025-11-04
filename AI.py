"""
AI-Powered Document Linking System
Uses multiple AI techniques: Embeddings, Semantic Analysis, Topic Modeling
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import openai  # Optional for GPT-based linking
import json

class AIDocumentLinker:
    """
    AI-powered document linking using:
    1. Sentence Transformers (BERT-based embeddings)
    2. Semantic similarity
    3. Topic modeling (LDA)
    4. Optional: GPT for relationship extraction
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', use_gpt=False, openai_key=None):
        """
        Initialize AI models
        
        Args:
            model_name: Sentence transformer model
                - 'all-MiniLM-L6-v2': Fast, good performance (default)
                - 'all-mpnet-base-v2': Better quality, slower
                - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual
            use_gpt: Whether to use GPT for enhanced linking
            openai_key: OpenAI API key if using GPT
        """
        print(f"Loading AI model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.use_gpt = use_gpt
        
        if use_gpt and openai_key:
            openai.api_key = openai_key
        
        print("✓ AI model loaded successfully")
    
    def extract_text_chunks(self, text, chunk_size=500):
        """Split text into semantic chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # Minimum chunk size
                chunks.append(chunk)
        
        return chunks
    
    def generate_embeddings(self, texts):
        """Generate semantic embeddings using transformer model"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def calculate_semantic_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between embeddings"""
        similarity = cosine_similarity(
            embedding1.reshape(1, -1), 
            embedding2.reshape(1, -1)
        )[0][0]
        return float(similarity)
    
    def extract_topics_lda(self, documents, n_topics=5):
        """Extract topics using Latent Dirichlet Allocation"""
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            max_df=0.8,
            min_df=2
        )
        
        try:
            doc_term_matrix = vectorizer.fit_transform(documents)
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42
            )
            lda.fit(doc_term_matrix)
            
            # Get topic distributions for each document
            topic_distributions = lda.transform(doc_term_matrix)
            
            # Get top words for each topic
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append(top_words)
            
            return topic_distributions, topics
        except:
            return None, None
    
    def analyze_with_gpt(self, text1, text2, filename1, filename2):
        """Use GPT to analyze relationship between documents"""
        if not self.use_gpt:
            return None
        
        prompt = f"""Analyze the relationship between these two documents:

Document 1 ({filename1}):
{text1[:1000]}...

Document 2 ({filename2}):
{text2[:1000]}...

Provide:
1. Main relationship type (e.g., "related concepts", "sequential", "contrasting views")
2. Key connecting themes (3-5 words)
3. Strength of connection (0-1 scale)

Return as JSON: {{"relationship_type": "", "themes": [], "strength": 0.0}}"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a document analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"GPT analysis error: {e}")
            return None
    
    def create_intelligent_links(self, documents, similarity_threshold=0.3):
        """
        Create intelligent links between documents using AI
        
        Args:
            documents: List of dicts with 'id', 'text', 'filename'
            similarity_threshold: Minimum similarity to create link
        
        Returns:
            List of links with AI-generated metadata
        """
        print(f"\n🤖 AI Analysis: Processing {len(documents)} documents...")
        
        # Generate embeddings for all documents
        print("1️⃣ Generating semantic embeddings...")
        texts = [doc['text'] for doc in documents]
        embeddings = self.generate_embeddings(texts)
        
        # Extract topics
        print("2️⃣ Extracting topics...")
        topic_distributions, topics = self.extract_topics_lda(texts)
        
        # Calculate similarities and create links
        print("3️⃣ Calculating semantic similarities...")
        links = []
        
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                doc1 = documents[i]
                doc2 = documents[j]
                
                # Calculate semantic similarity using embeddings
                semantic_sim = self.calculate_semantic_similarity(
                    embeddings[i], 
                    embeddings[j]
                )
                
                # Calculate topic similarity if available
                topic_sim = 0.0
                shared_topics = []
                if topic_distributions is not None:
                    topic_sim = cosine_similarity(
                        topic_distributions[i].reshape(1, -1),
                        topic_distributions[j].reshape(1, -1)
                    )[0][0]
                    
                    # Find dominant topics
                    dominant_topics_1 = np.argsort(topic_distributions[i])[-2:]
                    dominant_topics_2 = np.argsort(topic_distributions[j])[-2:]
                    shared = set(dominant_topics_1) & set(dominant_topics_2)
                    
                    if shared and topics:
                        for topic_idx in shared:
                            shared_topics.extend(topics[topic_idx][:3])
                
                # Combined AI score
                ai_score = (semantic_sim * 0.7) + (topic_sim * 0.3)
                
                # Optional: GPT analysis for high-similarity pairs
                gpt_analysis = None
                if self.use_gpt and ai_score > 0.5:
                    gpt_analysis = self.analyze_with_gpt(
                        doc1['text'], doc2['text'],
                        doc1['filename'], doc2['filename']
                    )
                
                # Create link if above threshold
                if ai_score >= similarity_threshold:
                    link = {
                        'source': doc1['id'],
                        'target': doc2['id'],
                        'semantic_similarity': round(float(semantic_sim), 3),
                        'topic_similarity': round(float(topic_sim), 3),
                        'ai_score': round(float(ai_score), 3),
                        'shared_topics': shared_topics[:5],
                        'link_type': self._classify_link_type(ai_score),
                        'confidence': self._calculate_confidence(semantic_sim, topic_sim)
                    }
                    
                    # Add GPT insights if available
                    if gpt_analysis:
                        link['relationship_type'] = gpt_analysis.get('relationship_type')
                        link['gpt_themes'] = gpt_analysis.get('themes', [])
                        link['gpt_strength'] = gpt_analysis.get('strength', 0.0)
                    
                    links.append(link)
        
        # Sort by AI score
        links.sort(key=lambda x: x['ai_score'], reverse=True)
        
        print(f"✓ Created {len(links)} intelligent connections\n")
        return links
    
    def _classify_link_type(self, score):
        """Classify link strength"""
        if score >= 0.7:
            return "strong"
        elif score >= 0.5:
            return "moderate"
        elif score >= 0.3:
            return "weak"
        else:
            return "minimal"
    
    def _calculate_confidence(self, semantic_sim, topic_sim):
        """Calculate confidence score"""
        # High confidence if both metrics agree
        if abs(semantic_sim - topic_sim) < 0.2:
            confidence = "high"
        elif abs(semantic_sim - topic_sim) < 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        return confidence
    
    def find_document_clusters(self, documents, n_clusters=3):
        """Group related documents into clusters"""
        from sklearn.cluster import KMeans
        
        texts = [doc['text'] for doc in documents]
        embeddings = self.generate_embeddings(texts)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(documents)), random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Organize by clusters
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'id': documents[idx]['id'],
                'filename': documents[idx]['filename']
            })
        
        return clusters
    
    def suggest_keywords_ai(self, text, top_n=10):
        """Extract keywords using AI embeddings"""
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        
        if not sentences:
            return []
        
        # Generate embeddings
        sentence_embeddings = self.generate_embeddings(sentences)
        
        # Calculate mean embedding (document representation)
        doc_embedding = np.mean(sentence_embeddings, axis=0)
        
        # Find most representative sentences
        similarities = cosine_similarity(
            sentence_embeddings,
            doc_embedding.reshape(1, -1)
        ).flatten()
        
        # Get top sentences
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        # Extract keywords from top sentences
        keywords = set()
        for idx in top_indices:
            words = sentences[idx].lower().split()
            keywords.update([w for w in words if len(w) > 4])
        
        return list(keywords)[:top_n]


# ============ EXAMPLE USAGE ============

if __name__ == "__main__":
    # Initialize AI linker
    ai_linker = AIDocumentLinker(
        model_name='all-MiniLM-L6-v2',  # Fast and efficient
        use_gpt=False  # Set True if you have OpenAI API key
    )
    
    # Sample documents (replace with your PDF texts)
    documents = [
        {
            'id': 'doc1.pdf',
            'filename': 'Machine Learning Intro',
            'text': 'Machine learning is a subset of artificial intelligence that focuses on training algorithms to learn from data. Neural networks are powerful models that can recognize patterns...'
        },
        {
            'id': 'doc2.pdf',
            'filename': 'Deep Learning Guide',
            'text': 'Deep learning uses neural networks with multiple layers. These deep neural networks can learn hierarchical representations of data, making them suitable for complex tasks...'
        },
        {
            'id': 'doc3.pdf',
            'filename': 'Data Science Basics',
            'text': 'Data science combines statistics, programming, and domain expertise to extract insights from data. Machine learning is an important tool in data science...'
        }
    ]
    
    # Create AI-powered links
    links = ai_linker.create_intelligent_links(
        documents, 
        similarity_threshold=0.3
    )
    
    # Display results
    print("=" * 60)
    print("AI-GENERATED LINKS")
    print("=" * 60)
    
    for link in links:
        print(f"\n{link['source']} ←→ {link['target']}")
        print(f"  AI Score: {link['ai_score']} ({link['link_type']} connection)")
        print(f"  Semantic Similarity: {link['semantic_similarity']}")
        print(f"  Topic Similarity: {link['topic_similarity']}")
        print(f"  Confidence: {link['confidence']}")
        if link['shared_topics']:
            print(f"  Shared Topics: {', '.join(link['shared_topics'][:3])}")
    
    # Find document clusters
    print("\n" + "=" * 60)
    print("DOCUMENT CLUSTERS")
    print("=" * 60)
    
    clusters = ai_linker.find_document_clusters(documents, n_clusters=2)
    for cluster_id, docs in clusters.items():
        print(f"\nCluster {cluster_id + 1}:")
        for doc in docs:
            print(f"  - {doc['filename']}")
    
    # Extract AI-suggested keywords
    print("\n" + "=" * 60)
    print("AI-SUGGESTED KEYWORDS")
    print("=" * 60)
    
    for doc in documents:
        keywords = ai_linker.suggest_keywords_ai(doc['text'], top_n=5)
        print(f"\n{doc['filename']}:")
        print(f"  {', '.join(keywords)}")
    
    # Save results
    output = {
        'links': links,
        'clusters': {str(k): v for k, v in clusters.items()}
    }
    
    with open('ai_mind_map.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n✅ Results saved to 'ai_mind_map.json'")
