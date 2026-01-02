"""
Medical Coding System - Base Agent
Foundation class for all AI agents with Azure OpenAI integration
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from openai import AzureOpenAI


# ============================================================================
# 1. IMPORT CONFIGURATION
# ============================================================================
from config.settings import AzureOpenAIConfig, AgentConfig
from config.prompts import get_prompt
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# 2. BASE AGENT CLASS
# ============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all AI agents
    Provides common functionality for LLM interaction, prompt management, and response parsing
    """
    
    def __init__(
        self,
        agent_name: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize base agent
        
        Args:
            agent_name: Name of the agent (used for logging)
            model: Azure OpenAI model to use (uses config default if None)
            temperature: Sampling temperature (uses config default if None)
            max_tokens: Maximum tokens in response (uses config default if None)
        """
        self.agent_name = agent_name
        self.model = model or AzureOpenAIConfig.CHAT_DEPLOYMENT
        self.temperature = temperature if temperature is not None else AzureOpenAIConfig.TEMPERATURE
        self.max_tokens = max_tokens or AzureOpenAIConfig.MAX_TOKENS
        
        # Initialize Azure OpenAI client
        self.client = self._initialize_client()
        
        # Agent statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_tokens': 0,
            'total_time': 0.0
        }
        
        logger.info(f"✓ {self.agent_name} initialized")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Temperature: {self.temperature}")
    
    def _initialize_client(self) -> AzureOpenAI:
        """Initialize Azure OpenAI client"""
        try:
            client = AzureOpenAI(
                api_key=AzureOpenAIConfig.API_KEY,
                api_version=AzureOpenAIConfig.API_VERSION,
                azure_endpoint=AzureOpenAIConfig.ENDPOINT
            )
            logger.info(f"  ✓ Azure OpenAI client initialized for {self.agent_name}")
            return client
        except Exception as e:
            logger.error(f"  ✗ Failed to initialize Azure OpenAI: {e}")
            raise
    
    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Call Azure OpenAI LLM
        
        Args:
            messages: List of message dictionaries
            temperature: Override default temperature
            max_tokens: Override default max tokens
            response_format: Optional response format (e.g., {"type": "json_object"})
        
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        self.stats['total_calls'] += 1
        
        try:
            # Prepare parameters
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens or self.max_tokens
            }
            
            # Add response format if specified
            if response_format:
                params["response_format"] = response_format
            
            # Make API call
            response = self.client.chat.completions.create(**params)
            
            duration = time.time() - start_time
            
            # Update statistics
            self.stats['successful_calls'] += 1
            self.stats['total_tokens'] += response.usage.total_tokens
            self.stats['total_time'] += duration
            
            # Extract response
            result = {
                'content': response.choices[0].message.content,
                'role': response.choices[0].message.role,
                'finish_reason': response.choices[0].finish_reason,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'duration': duration,
                'model': response.model
            }
            
            logger.info(f"{self.agent_name} LLM call successful "
                       f"({result['usage']['total_tokens']} tokens, {duration:.2f}s)")
            
            return result
        
        except Exception as e:
            self.stats['failed_calls'] += 1
            logger.error(f"{self.agent_name} LLM call failed: {e}")
            raise
    
    def _parse_json_response(self, content: str) -> Dict:
        """
        Parse JSON response from LLM
        
        Args:
            content: Response content string
        
        Returns:
            Parsed JSON dictionary
        """
        try:
            # Try direct JSON parse
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
                return json.loads(json_str)
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                json_str = content[start:end].strip()
                return json.loads(json_str)
            else:
                logger.error(f"Failed to parse JSON response: {content[:200]}...")
                raise
    
    def _retry_with_backoff(
        self,
        func,
        max_retries: Optional[int] = None,
        initial_delay: float = 1.0
    ):
        """
        Retry function with exponential backoff
        
        Args:
            func: Function to retry
            max_retries: Maximum retry attempts (uses config default if None)
            initial_delay: Initial delay in seconds
        
        Returns:
            Function result
        """
        max_retries = max_retries or AgentConfig.RETRY_ATTEMPTS
        delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All {max_retries} attempts failed")
                    raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        stats = self.stats.copy()
        
        # Calculate averages
        if stats['successful_calls'] > 0:
            stats['avg_tokens_per_call'] = stats['total_tokens'] / stats['successful_calls']
            stats['avg_time_per_call'] = stats['total_time'] / stats['successful_calls']
        else:
            stats['avg_tokens_per_call'] = 0
            stats['avg_time_per_call'] = 0
        
        return stats
    
    def reset_stats(self):
        """Reset agent statistics"""
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_tokens': 0,
            'total_time': 0.0
        }
        logger.info(f"{self.agent_name} statistics reset")
    
    @abstractmethod
    def process(self, input_data: Union[str, Dict]) -> Dict[str, Any]:
            """
            Process input data using prompt-based approach
            
            Args:
                input_data: String prompt or dict with 'prompt' and optional 'context'
            
            Returns:
                Processing results
            """
            if isinstance(input_data, str):
                # Simple string input
                return self.process_with_prompt(user_prompt=input_data)
            elif isinstance(input_data, dict):
                # Dictionary input with prompt and optional context
                return self.process_with_prompt(
                    user_prompt=input_data.get('prompt', ''),
                    context=input_data.get('context'),
                    parse_json=input_data.get('parse_json', False)
                )
            else:
                raise ValueError(f"Invalid input_data type: {type(input_data)}")
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"{self.__class__.__name__}(name='{self.agent_name}', "
                f"model='{self.model}', temperature={self.temperature})")


# ============================================================================
# 3. PROMPT-BASED AGENT (WITH TEMPLATE SUPPORT)
# ============================================================================

# ============================================================================
# 3. PROMPT-BASED AGENT (WITH TEMPLATE SUPPORT)
# ============================================================================

class PromptBasedAgent(BaseAgent):
    """
    Agent that uses prompt templates for structured interactions
    """
    
    def __init__(
        self,
        agent_name: str,
        prompt_template_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize prompt-based agent
        
        Args:
            agent_name: Name of the agent
            prompt_template_name: Name of prompt template to use
            system_prompt: System prompt (overrides template if provided)
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(agent_name, **kwargs)
        
        self.prompt_template_name = prompt_template_name
        self.system_prompt = system_prompt
        
        # Load prompt template if specified
        if prompt_template_name and not system_prompt:
            try:
                self.system_prompt = get_prompt(prompt_template_name)
                logger.info(f"  ✓ Loaded prompt template: {prompt_template_name}")
            except KeyError:
                logger.warning(f"  ⚠ Prompt template '{prompt_template_name}' not found")
    
    def create_messages(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Create message list for LLM
        
        Args:
            user_prompt: User/instruction prompt
            system_prompt: Override system prompt
            context: Additional context to include
        
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # System message
        sys_prompt = system_prompt or self.system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        
        # Context message (if provided)
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
        
        # User message
        messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    def process_with_prompt(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        parse_json: bool = False,
        **llm_kwargs
    ) -> Dict[str, Any]:
        """
        Process with prompt and return response
        
        Args:
            user_prompt: User prompt
            system_prompt: Override system prompt
            context: Additional context
            parse_json: Whether to parse response as JSON
            **llm_kwargs: Additional LLM parameters
        
        Returns:
            Processing results with response
        """
        # Create messages
        messages = self.create_messages(user_prompt, system_prompt, context)
        
        # Call LLM
        response = self._call_llm(messages, **llm_kwargs)
        
        # Parse response if requested
        if parse_json:
            try:
                parsed = self._parse_json_response(response['content'])
                response['parsed'] = parsed
            except Exception as e:
                logger.error(f"Failed to parse JSON response: {e}")
                response['parsed'] = None
        
        return response
    
    def process(self, input_data: Union[str, Dict]) -> Dict[str, Any]:
        """
        Process input data using prompt-based approach
        
        Args:
            input_data: String prompt or dict with 'prompt' and optional 'context'
        
        Returns:
            Processing results
        """
        if isinstance(input_data, str):
            # Simple string input
            return self.process_with_prompt(user_prompt=input_data)
        elif isinstance(input_data, dict):
            # Dictionary input with prompt and optional context
            return self.process_with_prompt(
                user_prompt=input_data.get('prompt', ''),
                context=input_data.get('context'),
                parse_json=input_data.get('parse_json', False)
            )
        else:
            raise ValueError(f"Invalid input_data type: {type(input_data)}")

# ============================================================================
# 4. MAIN BLOCK - Testing & Demonstration
# ============================================================================

if __name__ == "__main__":
    """
    Test and demonstrate base agent functionality
    Usage: python agents/base_agent.py
    """
    print("=" * 80)
    print("MEDICAL CODING SYSTEM - BASE AGENT TEST")
    print("=" * 80)
    print()
    
    # 1. Test BaseAgent with custom implementation
    print("🤖 Testing BaseAgent:")
    print("-" * 80)
    
    class TestAgent(BaseAgent):
        """Simple test agent"""
        
        def process(self, input_data: str) -> Dict[str, Any]:
            """Process test input"""
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_data}
            ]
            
            response = self._call_llm(messages, max_tokens=100)
            return {
                'input': input_data,
                'output': response['content'],
                'tokens': response['usage']['total_tokens']
            }
    
    # Initialize test agent
    test_agent = TestAgent(agent_name="TestAgent", temperature=0.3)
    print(f"✓ Created: {test_agent}")
    print()
    
    # Process test input
    print("Processing test query...")
    result = test_agent.process("What is medical coding?")
    print(f"✓ Response: {result['output'][:100]}...")
    print(f"  Tokens used: {result['tokens']}")
    print()
    
    # 2. Test PromptBasedAgent
    print("🤖 Testing PromptBasedAgent:")
    print("-" * 80)
    
    prompt_agent = PromptBasedAgent(
        agent_name="PromptAgent",
        system_prompt="You are a medical coding expert. Provide concise, accurate answers.",
        temperature=0.1
    )
    print(f"✓ Created: {prompt_agent}")
    print()
    
    # Test with prompt
    print("Processing medical query...")
    response = prompt_agent.process_with_prompt(
        user_prompt="What is ICD-10-CM code E11.9?",
        max_tokens=150
    )
    print(f"✓ Response: {response['content'][:200]}...")
    print(f"  Tokens: {response['usage']['total_tokens']}")
    print(f"  Duration: {response['duration']:.2f}s")
    print()
    
    # 3. Test JSON parsing
    print("🤖 Testing JSON response parsing:")
    print("-" * 80)
    
    json_response = prompt_agent.process_with_prompt(
        user_prompt="List 3 common ICD-10 diagnosis codes for diabetes in JSON format with 'codes' array containing objects with 'code' and 'description' fields.",
        parse_json=True,
        max_tokens=200
    )
    
    if json_response.get('parsed'):
        print("✓ Parsed JSON response:")
        print(json.dumps(json_response['parsed'], indent=2))
    else:
        print("✗ Failed to parse JSON")
        print(f"  Raw response: {json_response['content'][:200]}...")
    print()
    
    # 4. Test retry with backoff
    print("🤖 Testing retry mechanism:")
    print("-" * 80)
    
    class RetryTest:
        def __init__(self):
            self.attempt_count = 0
        
        def flaky_function(self):
            self.attempt_count += 1
            if self.attempt_count < 3:
                raise Exception(f"Simulated failure (attempt {self.attempt_count})")
            return "Success!"
    
    retry_test = RetryTest()
    
    try:
        result = test_agent._retry_with_backoff(retry_test.flaky_function, max_retries=5)
        print(f"✓ Retry successful after {retry_test.attempt_count} attempts: {result}")
    except Exception as e:
        print(f"✗ Retry failed: {e}")
    print()
    
    # 5. Test agent statistics
    print("📊 Agent Statistics:")
    print("-" * 80)
    
    test_stats = test_agent.get_stats()
    prompt_stats = prompt_agent.get_stats()
    
    print(f"TestAgent:")
    print(f"  Total calls: {test_stats['total_calls']}")
    print(f"  Successful: {test_stats['successful_calls']}")
    print(f"  Failed: {test_stats['failed_calls']}")
    print(f"  Total tokens: {test_stats['total_tokens']}")
    print(f"  Avg tokens/call: {test_stats['avg_tokens_per_call']:.1f}")
    print(f"  Avg time/call: {test_stats['avg_time_per_call']:.2f}s")
    print()
    
    print(f"PromptAgent:")
    print(f"  Total calls: {prompt_stats['total_calls']}")
    print(f"  Successful: {prompt_stats['successful_calls']}")
    print(f"  Total tokens: {prompt_stats['total_tokens']}")
    print(f"  Avg tokens/call: {prompt_stats['avg_tokens_per_call']:.1f}")
    print()
    
    # 6. Test context injection
    print("🤖 Testing context injection:")
    print("-" * 80)
    
    context = """
    Patient Information:
    - Age: 65 years
    - Chief Complaint: Shortness of breath
    - History: Type 2 diabetes, hypertension
    """
    
    response = prompt_agent.process_with_prompt(
        user_prompt="Based on the patient information, suggest relevant ICD-10 codes.",
        context=context.strip(),
        max_tokens=200
    )
    
    print(f"✓ Response with context: {response['content'][:250]}...")
    print()
    
    # 7. Summary
    print("=" * 80)
    print("✅ BASE AGENT TEST COMPLETE")
    print("=" * 80)
    print("\n💡 Usage Examples:")
    print()
    print("  from agents.base_agent import BaseAgent, PromptBasedAgent")
    print()
    print("  # Create custom agent")
    print("  class MyAgent(BaseAgent):")
    print("      def process(self, input_data):")
    print("          # Custom processing logic")
    print("          pass")
    print()
    print("  # Use prompt-based agent")
    print("  agent = PromptBasedAgent('MyAgent', system_prompt='...')")
    print("  result = agent.process_with_prompt('Query...')")
    print()