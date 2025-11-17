from typing import Dict, Optional, Generator
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
            "stream": True
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
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'response' in data:
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
3. If the user mentions "the article", "the paper" or "the document", assume they refer to the document that created the provided context
4. If the user asks questions about the title, name or authors of the article, use the document metadata and/or the first page of the document.
5. If the information is not in the context, clearly state: "I cannot find specific information about this in the document"
7. If there is contradictory or ambiguous information, mention it

DOCUMENT CONTEXT:
{context}

USER QUESTION: {prompt}
ANSWER (based on the context):"""
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
