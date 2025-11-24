from typing import Dict, Optional, Generator, List
import requests
import json

from .config import Config
from .utils import setup_logger


logger = setup_logger(__name__)


class OllamaClient:
    """
    Client for interacting with Ollama API for LLM inference.
    """
    
    def __init__(self, base_url: str = None, model: str = None):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Base URL for Ollama API
            model: Model name to use
        """
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.model = model or Config.OLLAMA_MODEL
        
        logger.info(f"Initialized OllamaClient with model: {self.model}")
        logger.info(f"Ollama URL: {self.base_url}")
    
    def check_connection(self) -> bool:
        """
        Check if Ollama server is accessible.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            return False
    
    def list_models(self) -> list:
        """
        List available models in Ollama.
        
        Returns:
            List of available model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    def get_models_detailed(self) -> Dict[str, list]:
        """
        Get available models categorized by type.
        
        Returns:
            Dictionary with 'text_generation', 'embedding', and 'vision' model lists
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return self._get_fallback_models()
            
            data = response.json()
            models = data.get('models', [])
            
            categorized = {
                'text_generation': [],
                'embedding': [],
                'vision': [],
                'memory': []
            }
            
            for model in models:
                name = model.get('name', '')
                model_lower = name.lower()
                
                # Categorize based on model name patterns
                if any(embed_indicator in model_lower for embed_indicator in 
                      ['embed', 'embedding', 'mxbai', 'nomic', 'bge', 'e5']):
                    categorized['embedding'].append(name)
                elif any(vision_indicator in model_lower for vision_indicator in 
                        ['vision', 'llava', 'bakllava', 'moondream']):
                    categorized['vision'].append(name)
                elif any(mem_indicator in model_lower for mem_indicator in
                        ['llama3.2', 'gemma', 'phi', 'qwen2', 'qwen3:4b']):
                    # Lighter models suitable for memory/clue generation
                    categorized['memory'].append(name)
                    categorized['text_generation'].append(name)
                else:
                    # Default to text generation
                    categorized['text_generation'].append(name)
            
            # Sort each category
            for category in categorized:
                categorized[category].sort()
            
            logger.info(f"Found {len(categorized['text_generation'])} text, "
                       f"{len(categorized['embedding'])} embedding, "
                       f"{len(categorized['vision'])} vision models")
            
            return categorized
            
        except Exception as e:
            logger.error(f"Error getting detailed models: {str(e)}")
            return self._get_fallback_models()
    
    def _get_fallback_models(self) -> Dict[str, list]:
        """
        Return fallback model configuration if Ollama is unavailable.
        
        Returns:
            Dictionary with default models from Config
        """
        return {
            'text_generation': [Config.OLLAMA_MODEL],
            'embedding': [Config.EMBEDDING_MODEL],
            'vision': [],
            'memory': [Config.MEMORY_MODEL]
        }
    
    def set_model(self, model_name: str) -> None:
        """
        Change the active model for generation.
        
        Args:
            model_name: Name of the model to use
        """
        self.model = model_name
        logger.info(f"Model changed to: {model_name}")
    
    def generate(self, 
                prompt: str, 
                context: str = "",
                temperature: float = 0.7,
                stream: bool = False) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: User prompt/question
            context: Additional context to include
            temperature: Sampling temperature (0.0 to 1.0)
            stream: Whether to stream the response
            
        Returns:
            Generated text response
        """
        # Construct full prompt with context
        full_prompt = self._build_prompt(prompt, context)
        
        logger.info(f"Generating response for prompt length: {len(full_prompt)}")
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "temperature": temperature,
            "stream": stream
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=stream,
                timeout=120
            )
            
            if stream:
                return self._handle_stream(response)
            else:
                if response.status_code == 200:
                    result = response.json().get('response', '')
                    logger.info(f"Generated response length: {len(result)}")
                    if not result:
                        logger.warning("Empty response from Ollama")
                        return "No se pudo generar una respuesta. Por favor, intenta de nuevo."
                    return result
                else:
                    error_msg = f"Error from Ollama: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return f"Error generating response: {response.status_code}"
                    
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_stream(self, 
                       prompt: str, 
                       context: str = "",
                       temperature: float = 0.7) -> Generator[str, None, None]:
        """
        Generate text with streaming response.
        
        Args:
            prompt: User prompt/question
            context: Additional context to include
            temperature: Sampling temperature
            
        Yields:
            Text chunks as they are generated
        """
        full_prompt = self._build_prompt(prompt, context)
        
        logger.info(f"Generating stream for prompt length: {len(full_prompt)}")
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "temperature": temperature,
            "stream": True,
            "options": {
                "num_predict": -1,  # Generate until natural end
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120
            )
            
            if response.status_code != 200:
                error_msg = f"Ollama returned status {response.status_code}: {response.text}"
                logger.error(error_msg)
                yield f"Error: {error_msg}"
                return
            
            chunk_count = 0
            # Use iter_lines with decode_unicode=True for immediate streaming
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        if 'response' in data and data['response']:
                            chunk_count += 1
                            yield data['response']
                        if data.get('done', False):
                            logger.info(f"Stream completed. Total chunks: {chunk_count}")
                            break
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {str(e)}, line: {line}")
                        continue
            
            if chunk_count == 0:
                logger.warning("No chunks received from Ollama")
                yield "No se generó respuesta. Por favor, intenta de nuevo."
                        
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            yield f"Error: {str(e)}"
    
    def _build_prompt(self, prompt: str, context: str = "") -> str:
        """
        Build the complete prompt with context.
        
        Args:
            prompt: User question
            context: Retrieved context from documents
            
        Returns:
            Formatted prompt string
        """
        if context:
            return f"""You are an expert academic assistant who helps answer questions about scientific documents.

CRITICAL INSTRUCTIONS:
1. Carefully READ all the provided context before answering
2. Base your answer ONLY on the information in the context
3. When you use information from the context, ALWAYS cite the page number inline using the format [pg. X] immediately after the information
4. Multiple sources can be cited together like [pg. 1, 3] or separately [pg. 1] ... [pg. 3]
5. JUST INCLUDE THE PAGE OF THE CITATION, DON'T INCLUDE THE DOCUMENT NUMBER.
6. If the user mentions "the article", "the paper" or "the document", assume they refer to the document that created the provided context
7. If the user asks questions about the title, name or authors of the article, use the document metadata and/or the first page of the document
8. If the information is not in the context, clearly state: "I cannot find specific information about this in the document"
9. If there is contradictory or ambiguous information, mention it
10. Be concise but complete. Start generating your response immediately.

CITATION EXAMPLES:
- "The Transformer architecture relies entirely on attention mechanisms [pg. 1]."
- "The model achieved state-of-the-art results [pg. 5] and outperformed previous approaches [pg. 6]."
- "Both experiments showed similar patterns [pg. 3, 7]."

DOCUMENT CONTEXT:
{context}

USER QUESTION: {prompt}

ANSWER (based on the context, with inline citations):"""
        else:
            return prompt
    
    def _handle_stream(self, response) -> str:
        """
        Handle streaming response and concatenate chunks.
        
        Args:
            response: Streaming response object
            
        Returns:
            Complete generated text
        """
        full_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if 'response' in data:
                    full_response += data['response']
        return full_response
